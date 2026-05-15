#!/usr/bin/env python3
"""
Exhaustive small-grid CTMC drift audit experiment.

This experiment directly supports Hypothesis H4 by exhaustively verifying
the Foster-Lyapunov drift inequality from
``docs/formal_math/z2/10_direct_ctmc_quadratic_proof_attempt.md``
on complete low-dimensional state grids.

What it does:
    1. Enumerates every integer state Q ∈ Z+^N with |Q|_1 ≤ max_norm
       for small toy systems (N=2,3).
    2. Computes the exact weighted-quadratic generator drift L V(Q) at
       every state.
    3. Computes the theorem RHS: -ε |Q|_1 + R.
    4. Verifies that L V(Q) ≤ -ε |Q|_1 + R at every state.
    5. Outputs all data as CSV (no figures).

Outputs:
    - Grid CSV:  system_id, state (JSON), |Q|_1, V(Q), LV(Q),
      theorem_rhs, residual, bound_holds.
    - Summary CSV: system_id, N, total_states, violations,
      max_residual, epsilon, R, status.
    - Human-readable markdown summary.

What it does not claim:
    - full benchmark-level stochastic certification by itself
    - anything beyond the declared toy grids

References:
    - z2/10_direct_ctmc_quadratic_proof_attempt.md  (Theorem 3)
    - z2/11_audit_of_direct_ctmc_quadratic_proof.md  (audit)
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
DEFAULT_MAX_NORM = 30


@dataclass(frozen=True)
class ToySystem:
    """Small system specification for exhaustive enumeration."""

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


def toy_systems() -> list[ToySystem]:
    """Return the toy system catalog for exhaustive audit."""
    return [
        ToySystem(
            system_id="toy_2server_symmetric",
            mu=(1.0, 1.0),
            lam=1.6,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
        ToySystem(
            system_id="toy_2server_asymmetric",
            mu=(1.0, 2.0),
            lam=2.4,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
        ToySystem(
            system_id="toy_3server",
            mu=(1.0, 1.5, 2.0),
            lam=3.6,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
        ToySystem(
            system_id="toy_2server_uas",
            mu=(1.0, 1.0),
            lam=1.6,
            alpha=10.0,
            beta=1.0,
            gamma=1.0,
            c=1.0,
        ),
    ]


# ──────────────────────────────────────────────────────────────────────
# State-grid enumeration
# ──────────────────────────────────────────────────────────────────────

def enumerate_states(N: int, max_norm: int) -> Iterator[np.ndarray]:
    """Enumerate all non-negative integer states with |Q|_1 ≤ max_norm.

    Parameters
    ----------
    N : int
        Dimension (number of servers).
    max_norm : int
        Maximum L1 norm.

    Yields
    ------
    np.ndarray
        Integer state vector of shape (N,).
    """
    for combo in itertools.product(range(max_norm + 1), repeat=N):
        state = np.array(combo, dtype=np.int64)
        if state.sum() <= max_norm:
            yield state


def count_states(N: int, max_norm: int) -> int:
    """Count the number of states in the grid (combinatorial)."""
    # |{Q ∈ Z+^N : |Q|_1 ≤ L}| = C(N+L, N)
    return math.comb(N + max_norm, N)


# ──────────────────────────────────────────────────────────────────────
# Theorem constants (from z2/10, §§4-7)
# ──────────────────────────────────────────────────────────────────────

def compute_theorem_constants(sys: ToySystem) -> dict[str, float]:
    """Compute the direct CTMC theorem constants ε and R.

    From z2/10_direct_ctmc_quadratic_proof_attempt.md §§4-7.

    Parameters
    ----------
    sys : ToySystem
        System specification.

    Returns
    -------
    dict
        Theorem constants including epsilon, R, C0, C1, kappa values.
    """
    mu = np.asarray(sys.mu, dtype=np.float64)
    mu_beta = np.power(mu, sys.beta)
    service_term = np.power(mu, 1.0 - sys.beta)
    kappa = sys.c / mu_beta - (sys.gamma / sys.alpha) * np.log(mu)

    C1 = float(
        math.log(sys.N) / sys.alpha + np.max(kappa) - np.min(kappa)
    )
    C0 = float(
        0.5 * sys.lam * np.max(np.power(mu, -sys.beta))
        + 0.5 * np.sum(service_term)
    )
    R = sys.lam * C1 + C0
    epsilon = min(
        (sys.Lambda - sys.lam) / float(np.sum(mu_beta)),
        float(np.min(service_term)),
    )

    return {
        "epsilon": epsilon,
        "R": R,
        "C0": C0,
        "C1": C1,
        "kappa": kappa.tolist(),
        "load_margin": sys.Lambda - sys.lam,
    }


# ──────────────────────────────────────────────────────────────────────
# Exact generator drift (from z2/10, §3)
# ──────────────────────────────────────────────────────────────────────

def routing_probabilities(
    Q: np.ndarray,
    *,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> np.ndarray:
    """Compute Reflected-UAS routing probabilities.

    Parameters
    ----------
    Q : np.ndarray
        Integer queue state.
    mu : np.ndarray
        Service rates.
    alpha, beta, gamma, c : float
        Policy parameters.

    Returns
    -------
    np.ndarray
        Probability vector, shape (N,).
    """
    log_w = np.log(np.power(mu, gamma)) - alpha * (Q.astype(np.float64) + c) / np.power(mu, beta)
    log_w -= np.max(log_w)  # Numerical stability
    w = np.exp(log_w)
    return w / np.sum(w)


def lyapunov_V(Q: np.ndarray, *, mu: np.ndarray, beta: float) -> float:
    """Evaluate the weighted-quadratic Lyapunov function V(Q).

    V(Q) = (1/2) Σ_i Q_i^2 / μ_i^β

    Parameters
    ----------
    Q : np.ndarray
        Integer queue state.
    mu : np.ndarray
        Service rates.
    beta : float
        Service-rate exponent.

    Returns
    -------
    float
        Value of V(Q).
    """
    q = Q.astype(np.float64)
    return float(0.5 * np.sum(q ** 2 / np.power(mu, beta)))


def exact_generator_drift(
    Q: np.ndarray,
    *,
    mu: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> float:
    """Compute the exact CTMC generator drift (L V)(Q).

    From z2/10_direct_ctmc_quadratic_proof_attempt.md §3:
        (L V)(Q) = λ Σ_i p_i(Q) (Q_i + 1/2) / μ_i^β
                 - Σ_i μ_i 1{Q_i>0} (Q_i - 1/2) / μ_i^β

    Parameters
    ----------
    Q : np.ndarray
        Integer queue state.
    mu, lam, alpha, beta, gamma, c : float
        System and policy parameters.

    Returns
    -------
    float
        Exact value of (L V)(Q).
    """
    q = Q.astype(np.float64)
    p = routing_probabilities(Q, mu=mu, alpha=alpha, beta=beta, gamma=gamma, c=c)
    mu_beta = np.power(mu, beta)

    arrival = lam * np.sum(p * (q + 0.5) / mu_beta)
    departure = np.sum(mu * (Q > 0).astype(np.float64) * (q - 0.5) / mu_beta)
    return float(arrival - departure)


# ──────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────

def run_exhaustive_audit(
    systems: list[ToySystem],
    output_dir: str | Path,
    *,
    max_norm: int = DEFAULT_MAX_NORM,
) -> tuple[Path, Path]:
    """Run the exhaustive drift audit.

    Parameters
    ----------
    systems : list of ToySystem
        Systems to audit.
    output_dir : str or Path
        Output directory.
    max_norm : int
        Maximum L1 norm for state enumeration.

    Returns
    -------
    tuple of Path
        Paths to (grid CSV, summary CSV).
    """
    # ── Grid CSV ──
    grid_columns = [
        Column("system_id", str, "System identifier"),
        Column("state", str, "Integer state Q (JSON)"),
        Column("Q_norm", int, "|Q|_1"),
        Column("V_Q", float, "V(Q) Lyapunov value"),
        Column("LV_Q", float, "(L V)(Q) generator drift"),
        Column("theorem_rhs", float, "-ε|Q|_1 + R"),
        Column("residual", float, "LV - theorem_rhs"),
        Column("bound_holds", bool, "LV ≤ theorem_rhs"),
    ]

    run_dir, _ = create_run_capsule(output_dir, "exhaustive_drift_audit")
    attach_run_log_handler(run_dir)
    run_metrics_dir = metrics_dir(run_dir)
    write_run_config(
        run_dir,
        {
            "experiment_name": "exhaustive_drift_audit",
            "output_dir": str(output_dir),
            "max_norm": max_norm,
            "system_ids": [sys.system_id for sys in systems],
        },
    )

    grid_writer = ExperimentCSVWriter(
        experiment_name="exhaustive_drift_grid",
        output_dir=run_metrics_dir,
        columns=grid_columns,
        metadata={
            "hypothesis": "H4",
            "theorem_reference": "z2/10_direct_ctmc_quadratic_proof_attempt.md",
            "max_norm": max_norm,
        },
    )

    # ── Summary CSV ──
    summary_columns = [
        Column("system_id", str, "System identifier"),
        Column("N", int, "Number of servers"),
        Column("lambda", float, "Arrival rate"),
        Column("Lambda", float, "Total capacity"),
        Column("rho", float, "Load factor"),
        Column("max_norm", int, "Max L1 norm for enumeration"),
        Column("total_states", int, "Total states enumerated"),
        Column("violations", int, "States where bound is violated"),
        Column("max_residual", float, "Max(LV - theorem_rhs) over all states"),
        Column("epsilon", float, "Theorem drift-rate constant ε"),
        Column("R", float, "Theorem remainder constant R"),
        Column("C0", float, "Constant C0"),
        Column("C1", float, "Constant C1"),
        Column("status", str, "PASS or FAIL"),
    ]

    summary_writer = ExperimentCSVWriter(
        experiment_name="exhaustive_drift_summary",
        output_dir=run_metrics_dir,
        columns=summary_columns,
        metadata={
            "hypothesis": "H4",
            "theorem_reference": "z2/10_direct_ctmc_quadratic_proof_attempt.md",
        },
    )

    summary_rows: list[dict[str, object]] = []

    for sys in iter_progress(
        systems,
        total=len(systems),
        desc="exhaustive drift",
    ):
        mu = np.asarray(sys.mu, dtype=np.float64)
        constants = compute_theorem_constants(sys)
        epsilon = constants["epsilon"]
        R = constants["R"]

        n_states = count_states(sys.N, max_norm)
        log.info(
            "Auditing %s (N=%d, load=%.4f): %d states, epsilon=%.6e, R=%.6e",
            sys.system_id, sys.N, sys.lam / sys.Lambda, n_states, epsilon, R,
        )

        violations = 0
        max_residual = -math.inf
        states_checked = 0

        for Q in enumerate_states(sys.N, max_norm):
            Q_norm = int(Q.sum())
            V_Q = lyapunov_V(Q, mu=mu, beta=sys.beta)
            LV_Q = exact_generator_drift(
                Q, mu=mu, lam=sys.lam,
                alpha=sys.alpha, beta=sys.beta,
                gamma=sys.gamma, c=sys.c,
            )
            rhs = -epsilon * Q_norm + R
            residual = LV_Q - rhs
            bound_ok = residual <= 1e-10

            if not bound_ok:
                violations += 1
            max_residual = max(max_residual, residual)

            grid_writer.write_row({
                "system_id": sys.system_id,
                "state": json.dumps(Q.tolist()),
                "Q_norm": Q_norm,
                "V_Q": V_Q,
                "LV_Q": LV_Q,
                "theorem_rhs": rhs,
                "residual": residual,
                "bound_holds": bound_ok,
            })
            states_checked += 1

        status = "PASS" if violations == 0 else "FAIL"
        log.info(
            "  %s: checked=%d, violations=%d, max_residual=%.6e, status=%s",
            sys.system_id, states_checked, violations, max_residual, status,
        )

        row = {
            "system_id": sys.system_id,
            "N": sys.N,
            "lambda": sys.lam,
            "Lambda": sys.Lambda,
            "rho": sys.lam / sys.Lambda,
            "max_norm": max_norm,
            "total_states": states_checked,
            "violations": violations,
            "max_residual": max_residual,
            "epsilon": epsilon,
            "R": R,
            "C0": constants["C0"],
            "C1": constants["C1"],
            "status": status,
        }
        summary_writer.write_row(row)
        summary_rows.append(row)

    grid_path = grid_writer.finalize()
    summary_path = summary_writer.finalize()
    report_path = metadata_path(run_dir, "exhaustive_drift_summary.md")
    lines = [
        "# Exhaustive Drift Audit Summary",
        "",
        "This report records exact weighted-quadratic drift checks on small toy",
        "grids. It does not by itself establish benchmark-level CTMC stability.",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['system_id']}: status={row['status']}, "
            f"violations={row['violations']}, "
            f"max_residual={row['max_residual']:.6e}, "
            f"epsilon={row['epsilon']:.6e}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return grid_path, summary_path


def configure_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Exhaustive small-grid CTMC drift audit (H4 support).",
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

    log.info("Starting exhaustive CTMC drift audit")
    systems = toy_systems()
    grid_path, summary_path = run_exhaustive_audit(
        systems, args.output_dir, max_norm=args.max_norm,
    )
    log.info("Grid data: %s", grid_path)
    log.info("Summary data: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
