#!/usr/bin/env python3
"""
CTMC generator boundary-mismatch demonstration experiment.

This experiment directly supports Hypothesis H3 by demonstrating
computationally that the old deterministic potential H does NOT
automatically certify the CTMC, because the generator drift
(L H)(Q) acquires a positive boundary term at states where Q_i = 0.

This is the computational evidence for the obstruction documented in:
    - ``docs/formal_math/z2/07_ctmc_scaling_gap.md``
    - ``docs/formal_math/z2/08_ctmc_generator_analysis.md``
    - ``docs/formal_math/z2/09_review_of_suggestions.md``

What it does:
    1. For small systems, enumerates states with at least one Q_i = 0
       (boundary states).
    2. Computes the EXACT CTMC generator drift (L H)(Q) using the exact
       one-step increment formulas (not Taylor approximations).
    3. Decomposes (L H)(Q) into:
       - interior_term: -Σ_{i: Q_i>0} μ_i^β (∂_i H)^2
       - boundary_term: Σ_{i: Q_i=0} λ p_i(Q) ∂_i H(Q)
       - remainder_term: R(Q)
    4. Identifies states where boundary_term > 0 (the obstruction).
    5. Compares against the reflected-ODE Lyapunov derivative, which
       clips boundary terms to zero.
    6. Outputs all data as CSV (no figures).

This experiment does NOT claim that H fails as a Foster-Lyapunov
function in all cases. It demonstrates the specific mechanism that
prevents a shortcut from the deterministic proof.

Outputs:
    - Per-state CSV: state, Q_norm, LH_exact, interior_term,
      boundary_term, boundary_positive, reflected_ode_dH, gap.
    - Summary CSV: total_boundary_states, positive_boundary_count,
      max_positive_boundary, max_gap.

What it does not claim:
    - global instability of the CTMC
    - failure of all possible CTMC Lyapunov arguments

References:
    - z2/08_ctmc_generator_analysis.md  (exact generator identity)
    - z2/07_ctmc_scaling_gap.md  (scaling gap)
    - z2/12_thesis_hypotheses.md  (H3)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gibbsq.qroute.utils.csv_writer import Column, ExperimentCSVWriter  # noqa: E402
from gibbsq.qroute.utils.run_artifacts import (  # noqa: E402
    attach_run_log_handler,
    create_run_capsule,
    metadata_path,
    metrics_dir,
    write_run_config,
)
from gibbsq.qroute.utils.progress import iter_progress  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "outputs/final"
DEFAULT_MAX_NORM = 25


@dataclass(frozen=True)
class SystemSpec:
    """Small system specification for boundary-mismatch audit."""

    system_id: str
    mu: tuple[float, ...]
    lam: float
    alpha: float
    beta: float
    gamma: float
    c: float

    @property
    def N(self) -> int:
        return len(self.mu)

    @property
    def Lambda(self) -> float:
        return sum(self.mu)


def demo_systems() -> list[SystemSpec]:
    """Return small systems for the boundary-mismatch demonstration."""
    return [
        SystemSpec(
            system_id="mismatch_2server_symmetric",
            mu=(1.0, 1.0),
            lam=1.6,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
        SystemSpec(
            system_id="mismatch_2server_asymmetric",
            mu=(1.0, 2.0),
            lam=2.4,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
        SystemSpec(
            system_id="mismatch_3server",
            mu=(1.0, 1.5, 2.0),
            lam=3.6,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
    ]


# ──────────────────────────────────────────────────────────────────────
# State enumeration (boundary states only)
# ──────────────────────────────────────────────────────────────────────

def enumerate_boundary_states(N: int, max_norm: int) -> Iterator[np.ndarray]:
    """Enumerate non-negative integer states with at least one Q_i = 0.

    Parameters
    ----------
    N : int
        Dimension.
    max_norm : int
        Maximum L1 norm.

    Yields
    ------
    np.ndarray
        Integer state vector with at least one zero coordinate.
    """
    for combo in itertools.product(range(max_norm + 1), repeat=N):
        state = np.array(combo, dtype=np.int64)
        if state.sum() <= max_norm and np.any(state == 0):
            yield state


def enumerate_all_states(N: int, max_norm: int) -> Iterator[np.ndarray]:
    """Enumerate all non-negative integer states with |Q|_1 ≤ max_norm."""
    for combo in itertools.product(range(max_norm + 1), repeat=N):
        state = np.array(combo, dtype=np.int64)
        if state.sum() <= max_norm:
            yield state


# ──────────────────────────────────────────────────────────────────────
# Exact CTMC generator for old potential H (from z2/08 §§2-3)
# ──────────────────────────────────────────────────────────────────────

def routing_probs(
    Q: np.ndarray,
    *,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> np.ndarray:
    """Compute reflected-UAS routing probabilities (numerically stable)."""
    log_w = gamma * np.log(mu) - alpha * (Q.astype(np.float64) + c) / np.power(mu, beta)
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / np.sum(w)


def potential_H(
    Q: np.ndarray,
    *,
    mu: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> float:
    """Evaluate the deterministic potential H(Q).

    From z2/05_projected_gradient_structure.md:
        H(Q) = Σ_i μ_i^{1-β} Q_i + (λ/α) log W(Q)
    """
    q = Q.astype(np.float64)
    log_w = gamma * np.log(mu) - alpha * (q + c) / np.power(mu, beta)
    log_W = np.max(log_w) + np.log(np.sum(np.exp(log_w - np.max(log_w))))
    return float(np.sum(np.power(mu, 1.0 - beta) * q) + (lam / alpha) * log_W)


def partial_H(
    Q: np.ndarray,
    *,
    mu: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> np.ndarray:
    """Compute ∂_i H(Q) = (μ_i - λ p_i(Q)) / μ_i^β.

    From z2/05_projected_gradient_structure.md §2.
    """
    p = routing_probs(Q, mu=mu, alpha=alpha, beta=beta, gamma=gamma, c=c)
    return (mu - lam * p) / np.power(mu, beta)


def exact_generator_LH(
    Q: np.ndarray,
    *,
    mu: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> float:
    """Compute the EXACT CTMC generator drift (L H)(Q).

    Uses the exact one-step increment formulas from z2/08 §2,
    NOT the Taylor approximation.

    (L H)(Q) = λ Σ_i p_i(Q) [H(Q+e_i) - H(Q)]
             + Σ_i μ_i 1{Q_i>0} [H(Q-e_i) - H(Q)]

    Parameters
    ----------
    Q : np.ndarray
        Integer queue state.
    mu, lam, alpha, beta, gamma, c : float
        System parameters.

    Returns
    -------
    float
        Exact value of (L H)(Q).
    """
    N = len(mu)
    p = routing_probs(Q, mu=mu, alpha=alpha, beta=beta, gamma=gamma, c=c)
    a = alpha / np.power(mu, beta)  # a_i = α / μ_i^β

    H_Q = potential_H(Q, mu=mu, lam=lam, alpha=alpha, beta=beta, gamma=gamma, c=c)
    result = 0.0

    for i in range(N):
        # Arrival increment: H(Q+e_i) - H(Q)
        Q_plus = Q.copy()
        Q_plus[i] += 1
        delta_plus = potential_H(Q_plus, mu=mu, lam=lam, alpha=alpha, beta=beta, gamma=gamma, c=c) - H_Q
        result += lam * p[i] * delta_plus

        # Service increment: H(Q-e_i) - H(Q), only if Q_i > 0
        if Q[i] > 0:
            Q_minus = Q.copy()
            Q_minus[i] -= 1
            delta_minus = potential_H(Q_minus, mu=mu, lam=lam, alpha=alpha, beta=beta, gamma=gamma, c=c) - H_Q
            result += mu[i] * delta_minus

    return result


def decompose_generator_drift(
    Q: np.ndarray,
    *,
    mu: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> dict[str, float]:
    """Decompose (L H)(Q) into interior, boundary, and remainder terms.

    From z2/08 §4:
        (L H)(Q) = -Σ_{i: Q_i>0} μ_i^β (∂_i H)^2
                 + Σ_{i: Q_i=0} λ p_i(Q) ∂_i H(Q)
                 + R(Q)

    where R(Q) is the second-order remainder.

    Parameters
    ----------
    Q : np.ndarray
        Integer queue state.

    Returns
    -------
    dict with keys:
        - LH_exact: exact generator drift
        - interior_term: -Σ μ_i^β (∂_i H)^2 for Q_i > 0
        - boundary_term: Σ λ p_i ∂_i H for Q_i = 0
        - remainder: LH_exact - interior_term - boundary_term
        - boundary_positive: whether boundary_term > 0
        - reflected_ode_dH: what the reflected-ODE Lyapunov derivative gives
        - gap: LH_exact - reflected_ode_dH
    """
    N = len(mu)
    p = routing_probs(Q, mu=mu, alpha=alpha, beta=beta, gamma=gamma, c=c)
    dH = partial_H(Q, mu=mu, lam=lam, alpha=alpha, beta=beta, gamma=gamma, c=c)

    # Exact CTMC generator drift
    LH_exact = exact_generator_LH(
        Q, mu=mu, lam=lam, alpha=alpha, beta=beta, gamma=gamma, c=c,
    )

    # Interior term: -Σ_{i: Q_i>0} μ_i^β (∂_i H)^2
    interior_term = 0.0
    for i in range(N):
        if Q[i] > 0:
            interior_term -= mu[i] ** beta * dH[i] ** 2

    # Boundary term: Σ_{i: Q_i=0} λ p_i ∂_i H
    boundary_term = 0.0
    for i in range(N):
        if Q[i] == 0:
            boundary_term += lam * p[i] * dH[i]

    # Remainder (second-order corrections)
    remainder = LH_exact - interior_term - boundary_term

    # Reflected-ODE Lyapunov derivative: clips boundary contributions to 0
    # For Q_i > 0: same as interior term
    # For Q_i = 0 with ∂_i H ≥ 0: clips to 0 (not positive contribution)
    # For Q_i = 0 with ∂_i H < 0: drift is inward, contributes negative
    reflected_ode_dH = interior_term
    for i in range(N):
        if Q[i] == 0 and dH[i] < 0:
            # Inward drift: contribute negative square term
            reflected_ode_dH -= mu[i] ** beta * dH[i] ** 2

    gap = LH_exact - reflected_ode_dH

    return {
        "LH_exact": LH_exact,
        "interior_term": interior_term,
        "boundary_term": boundary_term,
        "remainder": remainder,
        "boundary_positive": boundary_term > 1e-15,
        "reflected_ode_dH": reflected_ode_dH,
        "gap": gap,
    }


# ──────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────

def run_boundary_mismatch_demo(
    systems: list[SystemSpec],
    output_dir: str | Path,
    *,
    max_norm: int = DEFAULT_MAX_NORM,
) -> tuple[Path, Path]:
    """Run the boundary-mismatch demonstration experiment.

    Parameters
    ----------
    systems : list of SystemSpec
        Systems to demonstrate on.
    output_dir : str or Path
        Output directory.
    max_norm : int
        Maximum L1 norm for state enumeration.

    Returns
    -------
    tuple of Path
        Paths to (per-state CSV, summary CSV).
    """
    # ── Per-state CSV ──
    state_columns = [
        Column("system_id", str, "System identifier"),
        Column("state", str, "Integer state Q (JSON)"),
        Column("Q_norm", int, "|Q|_1"),
        Column("num_zero_coords", int, "Number of Q_i = 0"),
        Column("is_boundary", bool, "Whether state has at least one Q_i = 0"),
        Column("LH_exact", float, "Exact CTMC generator drift (L H)(Q)"),
        Column("interior_term", float, "-Σ μ^β (∂H)^2 for Q_i > 0"),
        Column("boundary_term", float, "Σ λ p_i ∂_i H for Q_i = 0"),
        Column("remainder", float, "Second-order remainder R(Q)"),
        Column("boundary_positive", bool, "Boundary term > 0 (the obstruction)"),
        Column("reflected_ode_dH", float, "Reflected-ODE Lyapunov derivative"),
        Column("gap", float, "LH_exact - reflected_ode_dH"),
    ]

    run_dir, _ = create_run_capsule(output_dir, "ctmc_boundary_mismatch_demo")
    attach_run_log_handler(run_dir)
    run_metrics_dir = metrics_dir(run_dir)
    write_run_config(
        run_dir,
        {
            "experiment_name": "ctmc_boundary_mismatch_demo",
            "output_dir": str(output_dir),
            "max_norm": max_norm,
            "system_ids": [spec.system_id for spec in systems],
        },
    )

    state_writer = ExperimentCSVWriter(
        experiment_name="ctmc_boundary_mismatch_states",
        output_dir=run_metrics_dir,
        columns=state_columns,
        metadata={
            "hypothesis": "H3",
            "theorem_reference": "z2/08_ctmc_generator_analysis.md",
            "description": (
                "Demonstrates the exact CTMC generator boundary mismatch "
                "that prevents the old deterministic potential H from "
                "automatically certifying the CTMC."
            ),
            "max_norm": max_norm,
        },
    )

    # ── Summary CSV ──
    summary_columns = [
        Column("system_id", str, "System identifier"),
        Column("N", int, "Number of servers"),
        Column("lambda", float, "Arrival rate"),
        Column("rho", float, "Load factor"),
        Column("max_norm", int, "Max L1 norm"),
        Column("total_states", int, "Total states enumerated"),
        Column("boundary_states", int, "States with at least one Q_i = 0"),
        Column("positive_boundary_count", int, "Boundary states with positive boundary term"),
        Column("positive_boundary_fraction", float, "Fraction of boundary states with obstruction"),
        Column("max_positive_boundary_term", float, "Max positive boundary term observed"),
        Column("max_gap", float, "Max gap between CTMC drift and reflected-ODE drift"),
        Column("mean_gap_at_boundary", float, "Mean gap at boundary states"),
        Column("obstruction_demonstrated", bool, "Whether positive boundary terms exist"),
    ]

    summary_writer = ExperimentCSVWriter(
        experiment_name="ctmc_boundary_mismatch_summary",
        output_dir=run_metrics_dir,
        columns=summary_columns,
        metadata={
            "hypothesis": "H3",
            "theorem_reference": "z2/08_ctmc_generator_analysis.md",
        },
    )

    summary_rows: list[dict[str, object]] = []

    for spec in iter_progress(
        systems,
        total=len(systems),
        desc="boundary mismatch",
    ):
        mu = np.asarray(spec.mu, dtype=np.float64)
        rho = spec.lam / spec.Lambda
        log.info(
            "Boundary-mismatch demo: %s (N=%d, load=%.4f)",
            spec.system_id, spec.N, rho,
        )

        total_states = 0
        boundary_states = 0
        positive_boundary_count = 0
        max_positive_boundary_term = 0.0
        max_gap = -math.inf
        gap_sum_boundary = 0.0

        for Q in enumerate_all_states(spec.N, max_norm):
            total_states += 1
            Q_norm = int(Q.sum())
            num_zero = int(np.sum(Q == 0))
            is_boundary = num_zero > 0

            decomp = decompose_generator_drift(
                Q, mu=mu, lam=spec.lam, alpha=spec.alpha,
                beta=spec.beta, gamma=spec.gamma, c=spec.c,
            )

            if is_boundary:
                boundary_states += 1
                gap_sum_boundary += decomp["gap"]
                if decomp["boundary_positive"]:
                    positive_boundary_count += 1
                    max_positive_boundary_term = max(
                        max_positive_boundary_term, decomp["boundary_term"]
                    )
                max_gap = max(max_gap, decomp["gap"])

            state_writer.write_row({
                "system_id": spec.system_id,
                "state": json.dumps(Q.tolist()),
                "Q_norm": Q_norm,
                "num_zero_coords": num_zero,
                "is_boundary": is_boundary,
                "LH_exact": decomp["LH_exact"],
                "interior_term": decomp["interior_term"],
                "boundary_term": decomp["boundary_term"],
                "remainder": decomp["remainder"],
                "boundary_positive": decomp["boundary_positive"],
                "reflected_ode_dH": decomp["reflected_ode_dH"],
                "gap": decomp["gap"],
            })

        mean_gap = gap_sum_boundary / boundary_states if boundary_states > 0 else 0.0
        obstruction = positive_boundary_count > 0

        log.info(
            "  %s: total=%d, boundary=%d, positive_boundary=%d (%.1f%%), "
            "max_boundary_term=%.6e, max_gap=%.6e, obstruction=%s",
            spec.system_id, total_states, boundary_states,
            positive_boundary_count,
            100.0 * positive_boundary_count / boundary_states if boundary_states > 0 else 0.0,
            max_positive_boundary_term, max_gap, obstruction,
        )

        row = {
            "system_id": spec.system_id,
            "N": spec.N,
            "lambda": spec.lam,
            "rho": rho,
            "max_norm": max_norm,
            "total_states": total_states,
            "boundary_states": boundary_states,
            "positive_boundary_count": positive_boundary_count,
            "positive_boundary_fraction": (
                positive_boundary_count / boundary_states
                if boundary_states > 0 else 0.0
            ),
            "max_positive_boundary_term": max_positive_boundary_term,
            "max_gap": max_gap if max_gap > -math.inf else 0.0,
            "mean_gap_at_boundary": mean_gap,
            "obstruction_demonstrated": obstruction,
        }
        summary_writer.write_row(row)
        summary_rows.append(row)

    state_path = state_writer.finalize()
    summary_path = summary_writer.finalize()
    report_path = metadata_path(run_dir, "ctmc_boundary_mismatch_summary.md")
    lines = [
        "# CTMC Boundary Mismatch Summary",
        "",
        "This report demonstrates the specific boundary obstruction in the old",
        "deterministic-potential shortcut. It does not claim global CTMC",
        "instability or rule out other Lyapunov routes.",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['system_id']}: obstruction_demonstrated={row['obstruction_demonstrated']}, "
            f"positive_boundary_count={row['positive_boundary_count']}, "
            f"max_positive_boundary_term={row['max_positive_boundary_term']:.6e}, "
            f"max_gap={row['max_gap']:.6e}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return state_path, summary_path


def configure_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "CTMC generator boundary-mismatch demonstration (H3 support). "
            "Shows computationally that the old potential H acquires a "
            "positive boundary term in the CTMC generator."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-norm", type=int, default=DEFAULT_MAX_NORM)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    log.info("Starting CTMC boundary-mismatch demonstration")
    systems = demo_systems()
    state_path, summary_path = run_boundary_mismatch_demo(
        systems, args.output_dir, max_norm=args.max_norm,
    )
    log.info("State data: %s", state_path)
    log.info("Summary data: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
