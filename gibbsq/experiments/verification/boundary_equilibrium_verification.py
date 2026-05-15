#!/usr/bin/env python3
"""
Exact boundary-equilibrium verification experiment.

This experiment directly supports Hypothesis H2 and validates the
closed-form equilibrium characterization from
``docs/formal_math/z2/02_boundary_equilibrium.md``.

What it does:
    1. Computes the scalar equilibrium solution K* for the benchmark
       system and additional auxiliary systems.
    2. Reconstructs the explicit equilibrium vector q* from the
       closed-form formula.
    3. Runs the reflected-ODE numerical integrator and compares the
       attractor against the closed-form prediction.
    4. Outputs all results as CSV data (no figures).

Outputs:
    - CSV with columns: system_id, N, lambda, Lambda, rho, K_star,
      q_star (JSON), active_set (JSON), max_discrepancy,
      active_set_match, etc.
    - Human-readable markdown summary.

What it does not claim:
    - stochastic stability of the CTMC
    - anything beyond deterministic equilibrium consistency

References:
    - z2/02_boundary_equilibrium.md  (Theorem 1)
    - z2/12_thesis_hypotheses.md     (Hypothesis H2)
    - z2/13_thesis_evidence_requirements.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import brentq

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gibbsq.qroute.utils.csv_writer import Column, ExperimentCSVWriter  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "outputs/data"
DEFAULT_CONFIG_NAME = "final_experiment"

# ──────────────────────────────────────────────────────────────────────
# Benchmark system definitions
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SystemSpec:
    """Specification for a queueing system under Reflected UAS.

    Parameters
    ----------
    system_id : str
        Unique identifier for this system specification.
    mu : tuple[float, ...]
        Service rates for each server.
    lam : float
        Arrival rate.
    alpha : float
        Softmax inverse-temperature parameter.
    beta : float
        Service-rate exponent in the routing weights.
    gamma : float
        Service-rate prefactor exponent.
    c : float
        Queue-offset constant.
    """

    system_id: str
    mu: tuple[float, ...]
    lam: float
    alpha: float
    beta: float
    gamma: float
    c: float

    @property
    def N(self) -> int:
        """Number of servers."""
        return len(self.mu)

    @property
    def Lambda(self) -> float:
        """Total service capacity."""
        return sum(self.mu)

    @property
    def rho(self) -> float:
        """System load factor."""
        return self.lam / self.Lambda


def _load_profile_system(config_name: str) -> SystemSpec:
    raw_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / f"{config_name}.yaml")
    service_rates = tuple(float(x) for x in raw_cfg.system.service_rates)
    arrival_rate = float(raw_cfg.system.arrival_rate)
    return SystemSpec(
        system_id=f"{config_name}_benchmark",
        mu=service_rates,
        lam=arrival_rate,
        alpha=20.0,
        beta=0.85,
        gamma=0.5,
        c=0.5,
    )


def benchmark_systems(config_name: str = DEFAULT_CONFIG_NAME) -> list[SystemSpec]:
    """Return the declared benchmark system catalog.

    This catalog includes the primary 10-server benchmark used in the
    thesis and several auxiliary systems for multi-scale validation.
    """
    benchmark = _load_profile_system(config_name)
    return [
        benchmark,
        SystemSpec(
            system_id="symmetric_4server",
            mu=(1.0, 1.0, 1.0, 1.0),
            lam=3.2,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
        SystemSpec(
            system_id="heavy_load_5server",
            mu=(1.0, 1.5, 2.0, 2.5, 3.0),
            lam=9.5,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
        SystemSpec(
            system_id="light_load_3server",
            mu=(2.0, 3.0, 5.0),
            lam=2.0,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
        SystemSpec(
            system_id="uas_special_case",
            mu=benchmark.mu,
            lam=benchmark.lam,
            alpha=10.0,
            beta=1.0,
            gamma=1.0,
            c=1.0,
        ),
    ]


# ──────────────────────────────────────────────────────────────────────
# Mathematical core: implements z2/02_boundary_equilibrium.md
# ──────────────────────────────────────────────────────────────────────

def compute_theta(spec: SystemSpec) -> np.ndarray:
    """Compute the threshold constants θ_i.

    From z2/02_boundary_equilibrium.md §1:
        θ_i = μ_i^{γ-1} exp(-α c / μ_i^β)

    Parameters
    ----------
    spec : SystemSpec
        System specification.

    Returns
    -------
    np.ndarray
        Array of threshold constants, shape (N,).
    """
    mu = np.asarray(spec.mu, dtype=np.float64)
    return np.power(mu, spec.gamma - 1.0) * np.exp(
        -spec.alpha * spec.c / np.power(mu, spec.beta)
    )


def stable_policy_probs(
    q: np.ndarray,
    *,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> np.ndarray:
    """Return numerically stable Reflected-UAS routing probabilities."""
    mu_beta = np.power(mu, beta)
    log_w = gamma * np.log(mu) - alpha * (q + c) / mu_beta
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    W = np.sum(w)
    if W <= 0.0 or not np.isfinite(W):
        return np.full_like(mu, 1.0 / len(mu), dtype=np.float64)
    return w / W


def G_function(K: float, mu: np.ndarray, theta: np.ndarray) -> float:
    """Evaluate the scalar consistency function G(K).

    From z2/02_boundary_equilibrium.md §4:
        G(K) = Σ_i min(μ_i, μ_i θ_i / K)

    Parameters
    ----------
    K : float
        Scalar parameter (must be > 0).
    mu : np.ndarray
        Service rates.
    theta : np.ndarray
        Threshold constants.

    Returns
    -------
    float
        Value of G(K).
    """
    if K <= 0.0:
        raise ValueError(f"K must be strictly positive, got {K}")
    return float(np.sum(np.minimum(mu, mu * theta / K)))


def solve_K_star(spec: SystemSpec) -> float:
    """Solve the scalar equilibrium equation λ = G(K*).

    Uses Brent's method on the monotone function G(K) to find the
    unique root.  See z2/02_boundary_equilibrium.md, Proposition 2.

    Parameters
    ----------
    spec : SystemSpec
        System specification.

    Returns
    -------
    float
        The unique equilibrium parameter K*.

    Raises
    ------
    ValueError
        If the system violates the stability condition λ < Λ.
    """
    mu = np.asarray(spec.mu, dtype=np.float64)
    theta = compute_theta(spec)

    if spec.lam >= spec.Lambda:
        raise ValueError(
            f"System '{spec.system_id}' violates λ < Λ: "
            f"λ={spec.lam}, Λ={spec.Lambda}"
        )

    # G is monotone decreasing.  Find bracket [K_lo, K_hi].
    # G(K) → Λ as K → 0+  and  G(K) → 0 as K → ∞.
    K_lo = 1e-30
    K_hi = float(np.max(mu * theta)) * 100.0

    # Verify bracket
    g_lo = G_function(K_lo, mu, theta)
    g_hi = G_function(K_hi, mu, theta)
    if not (g_lo > spec.lam > g_hi):
        raise RuntimeError(
            f"Bracket failure for '{spec.system_id}': "
            f"G({K_lo})={g_lo}, λ={spec.lam}, G({K_hi})={g_hi}"
        )

    K_star = brentq(
        lambda K: G_function(K, mu, theta) - spec.lam,
        K_lo,
        K_hi,
        xtol=1e-15,
        rtol=1e-15,
    )
    return float(K_star)


def compute_equilibrium(spec: SystemSpec, K_star: float) -> np.ndarray:
    """Compute the equilibrium vector q* from the scalar K*.

    From z2/02_boundary_equilibrium.md §3:
        q*_i = max(0, (μ_i^β / α) log(θ_i / K*))

    Parameters
    ----------
    spec : SystemSpec
        System specification.
    K_star : float
        Scalar equilibrium parameter.

    Returns
    -------
    np.ndarray
        Equilibrium vector q*, shape (N,).
    """
    mu = np.asarray(spec.mu, dtype=np.float64)
    theta = compute_theta(spec)
    q_star = np.maximum(
        0.0,
        np.power(mu, spec.beta) / spec.alpha * np.log(theta / K_star),
    )
    return q_star


def compute_active_set(spec: SystemSpec, K_star: float) -> list[int]:
    """Compute the active set A* = {i : K* < θ_i}.

    Parameters
    ----------
    spec : SystemSpec
        System specification.
    K_star : float
        Scalar equilibrium parameter.

    Returns
    -------
    list of int
        Indices of servers with positive equilibrium queue lengths.
    """
    theta = compute_theta(spec)
    return [int(i) for i in range(spec.N) if K_star < theta[i]]


def verify_equilibrium_conditions(
    spec: SystemSpec,
    q_star: np.ndarray,
) -> dict[str, float]:
    """Verify the equilibrium complementarity conditions.

    Checks the conditions from z2/01_reflected_fluid_model.md §6:
        q*_i ≥ 0
        λ p_i(q*) ≤ μ_i
        q*_i (μ_i - λ p_i(q*)) = 0

    Parameters
    ----------
    spec : SystemSpec
        System specification.
    q_star : np.ndarray
        Candidate equilibrium vector.

    Returns
    -------
    dict
        Verification metrics including max violations.
    """
    mu = np.asarray(spec.mu, dtype=np.float64)

    p = stable_policy_probs(
        q_star,
        mu=mu,
        alpha=spec.alpha,
        beta=spec.beta,
        gamma=spec.gamma,
        c=spec.c,
    )

    # Check conditions
    lam_p = spec.lam * p
    nonneg_violation = float(np.max(np.maximum(0.0, -q_star)))
    capacity_violation = float(np.max(np.maximum(0.0, lam_p - mu)))
    complementarity_residual = float(np.max(np.abs(q_star * (mu - lam_p))))

    return {
        "max_nonnegativity_violation": nonneg_violation,
        "max_capacity_violation": capacity_violation,
        "max_complementarity_residual": complementarity_residual,
    }


# ──────────────────────────────────────────────────────────────────────
# Numerical ODE attractor (for cross-validation)
# ──────────────────────────────────────────────────────────────────────

def simulate_reflected_ode(
    spec: SystemSpec,
    q0: np.ndarray | None = None,
    *,
    dt: float = 0.01,
    max_time: float = 500.0,
) -> np.ndarray:
    """Simulate the reflected ODE to find the numerical attractor.

    Uses forward-Euler integration with explicit orthant projection
    (the standard Skorokhod map for the nonneg orthant).

    Parameters
    ----------
    spec : SystemSpec
        System specification.
    q0 : np.ndarray, optional
        Initial state.  Defaults to a scaled identity vector.
    dt : float
        Integration time step.
    max_time : float
        Maximum integration time.

    Returns
    -------
    np.ndarray
        Terminal state (numerical attractor), shape (N,).
    """
    mu = np.asarray(spec.mu, dtype=np.float64)
    N = spec.N

    if q0 is None:
        q0 = np.ones(N, dtype=np.float64) * 5.0

    q = q0.copy()
    n_steps = int(max_time / dt)

    for _ in range(n_steps):
        p = stable_policy_probs(
            q,
            mu=mu,
            alpha=spec.alpha,
            beta=spec.beta,
            gamma=spec.gamma,
            c=spec.c,
        )

        # Drift
        drift = spec.lam * p - mu

        # Forward Euler + orthant projection (Skorokhod reflection)
        q = np.maximum(0.0, q + dt * drift)

    return q


# ──────────────────────────────────────────────────────────────────────
# Main experiment logic
# ──────────────────────────────────────────────────────────────────────

def run_verification(
    systems: list[SystemSpec],
    output_dir: str | Path,
) -> Path:
    """Run the full boundary-equilibrium verification experiment.

    Parameters
    ----------
    systems : list of SystemSpec
        Systems to verify.
    output_dir : str or Path
        Output directory for CSV data.

    Returns
    -------
    Path
        Path to the generated CSV file.
    """
    columns = [
        Column("system_id", str, "Unique system identifier"),
        Column("N", int, "Number of servers"),
        Column("lambda", float, "Arrival rate"),
        Column("Lambda", float, "Total service capacity"),
        Column("rho", float, "System load factor"),
        Column("alpha", float, "Softmax inverse-temperature"),
        Column("beta", float, "Service-rate exponent"),
        Column("gamma", float, "Prefactor exponent"),
        Column("c", float, "Queue-offset constant"),
        Column("K_star", float, "Scalar equilibrium parameter K*"),
        Column("q_star", str, "Equilibrium vector q* (JSON array)"),
        Column("active_set", str, "Active set A* (JSON array)"),
        Column("active_set_size", int, "Number of servers with q*_i > 0"),
        Column("q_star_sum", float, "Sum of equilibrium queue lengths"),
        Column("numerical_attractor", str, "ODE numerical attractor (JSON array)"),
        Column("max_discrepancy", float, "Max |q*_exact - q*_numerical|"),
        Column("active_set_match", bool, "Whether active sets agree"),
        Column("max_nonnegativity_violation", float, "Max violation of q*_i >= 0"),
        Column("max_capacity_violation", float, "Max violation of λp_i ≤ μ_i"),
        Column("max_complementarity_residual", float, "Max |q*_i(μ_i - λp_i)|"),
        Column("scalar_equation_residual", float, "|λ - G(K*)|"),
        Column("status", str, "PASS or FAIL"),
    ]

    writer = ExperimentCSVWriter(
        experiment_name="boundary_equilibrium_verification",
        output_dir=output_dir,
        columns=columns,
        metadata={
            "hypothesis": "H2",
            "theorem_reference": "z2/02_boundary_equilibrium.md",
            "description": (
                "Exact boundary-equilibrium verification.  Computes K* via "
                "Brent's method, reconstructs q*, and cross-validates against "
                "reflected-ODE numerical attractor."
            ),
        },
    )

    summary_rows: list[dict[str, object]] = []

    for spec in systems:
        log.info("Verifying system: %s (N=%d, ρ=%.4f)", spec.system_id, spec.N, spec.rho)

        # Step 1: Solve scalar equation
        K_star = solve_K_star(spec)
        log.info("  K* = %.15e", K_star)

        # Step 2: Compute exact equilibrium
        q_star = compute_equilibrium(spec, K_star)
        active_set = compute_active_set(spec, K_star)
        log.info("  q* = %s", np.array2string(q_star, precision=8))
        log.info("  Active set = %s", active_set)

        # Step 3: Verify complementarity conditions
        cond = verify_equilibrium_conditions(spec, q_star)

        # Step 4: Compute scalar equation residual
        mu = np.asarray(spec.mu, dtype=np.float64)
        theta = compute_theta(spec)
        scalar_residual = abs(spec.lam - G_function(K_star, mu, theta))

        # Step 5: Cross-validate against numerical ODE attractor
        numerical_attractor = simulate_reflected_ode(spec)
        max_discrepancy = float(np.max(np.abs(q_star - numerical_attractor)))

        # Active set comparison
        numerical_active = [int(i) for i in range(spec.N) if numerical_attractor[i] > 1e-6]
        active_match = set(active_set) == set(numerical_active)

        # Determine pass/fail
        TOLERANCE = 1e-4
        status = "PASS"
        if max_discrepancy > TOLERANCE:
            status = "FAIL"
            log.warning("  FAIL: max_discrepancy=%.6e > tolerance=%.6e", max_discrepancy, TOLERANCE)
        if not active_match:
            status = "FAIL"
            log.warning("  FAIL: active set mismatch: exact=%s, numerical=%s", active_set, numerical_active)
        if cond["max_complementarity_residual"] > 1e-8:
            status = "FAIL"
            log.warning("  FAIL: complementarity residual=%.6e", cond["max_complementarity_residual"])

        log.info("  max_discrepancy = %.6e, status = %s", max_discrepancy, status)

        row = {
            "system_id": spec.system_id,
            "N": spec.N,
            "lambda": spec.lam,
            "Lambda": spec.Lambda,
            "rho": spec.rho,
            "alpha": spec.alpha,
            "beta": spec.beta,
            "gamma": spec.gamma,
            "c": spec.c,
            "K_star": K_star,
            "q_star": json.dumps(q_star.tolist()),
            "active_set": json.dumps(active_set),
            "active_set_size": len(active_set),
            "q_star_sum": float(np.sum(q_star)),
            "numerical_attractor": json.dumps(numerical_attractor.tolist()),
            "max_discrepancy": max_discrepancy,
            "active_set_match": active_match,
            "max_nonnegativity_violation": cond["max_nonnegativity_violation"],
            "max_capacity_violation": cond["max_capacity_violation"],
            "max_complementarity_residual": cond["max_complementarity_residual"],
            "scalar_equation_residual": scalar_residual,
            "status": status,
        }
        writer.write_row(row)
        summary_rows.append(row)

    csv_path = writer.finalize()
    summary_path = csv_path.with_name("boundary_equilibrium_verification_summary.md")
    summary_lines = [
        "# Boundary Equilibrium Verification Summary",
        "",
        "This report documents deterministic agreement between the closed-form",
        "boundary equilibrium and the reflected-ODE attractor. It does not",
        "claim stochastic CTMC stability.",
        "",
    ]
    for row in summary_rows:
        summary_lines.append(
            f"- {row['system_id']}: status={row['status']}, "
            f"K*={row['K_star']:.12e}, "
            f"max_discrepancy={row['max_discrepancy']:.6e}, "
            f"active_set_match={row['active_set_match']}"
        )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return csv_path


def configure_logging() -> None:
    """Configure structured logging for the experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Exact boundary-equilibrium verification experiment.  "
            "Validates Hypothesis H2 from the z2 thesis program."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CSV data files.",
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the boundary-equilibrium verification experiment."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    log.info("Starting boundary-equilibrium verification experiment")
    systems = benchmark_systems(args.config_name)
    csv_path = run_verification(systems, args.output_dir)
    log.info("Results written to: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
