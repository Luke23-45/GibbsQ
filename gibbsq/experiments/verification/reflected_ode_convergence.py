#!/usr/bin/env python3
"""
Reflected-ODE multi-start convergence verification experiment.

This experiment directly supports Hypotheses H1 and H2 by validating the
global convergence theorem from
``docs/formal_math/z2/06_global_convergence_reflected_ode.md``.

What it does:
    1. Runs the reflected-ODE integrator from a diverse bank of initial
       states (random, extreme, boundary-adjacent).
    2. Verifies that every trajectory converges to the same terminal
       equilibrium neighbourhood.
    3. Measures terminal pairwise diameter, max equilibrium-residual
       norm, and Lyapunov-function descent monotonicity.
    4. Outputs all results as CSV data (no figures).

Outputs:
    - Per-trajectory CSV: system_id, ic_family, ic_index, terminal_state,
      terminal_residual_norm, H_value, converged, etc.
    - Summary CSV: system_id, terminal_diameter, max_residual_norm,
      H_monotone_fraction, num_trajectories, etc.

References:
    - z2/01_reflected_fluid_model.md  (reflected ODE definition)
    - z2/05_projected_gradient_structure.md  (potential H)
    - z2/06_global_convergence_reflected_ode.md  (Theorem 2)
    - z2/12_thesis_hypotheses.md  (H1, H2)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gibbsq.qroute.utils.csv_writer import Column, ExperimentCSVWriter  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "outputs/data"
DEFAULT_DT = 0.01
DEFAULT_MAX_TIME = 800.0
DEFAULT_H_SAMPLE_INTERVAL = 1.0
CONVERGENCE_TOL = 1e-5


@dataclass(frozen=True)
class SystemSpec:
    """Queueing system specification."""

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

    @property
    def rho(self) -> float:
        return self.lam / self.Lambda


def benchmark_systems() -> list[SystemSpec]:
    """Return the benchmark system catalog."""
    return [
        SystemSpec(
            system_id="benchmark_10server",
            mu=tuple(0.5 + 0.2 * i for i in range(10)),
            lam=11.2,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
        SystemSpec(
            system_id="symmetric_4server",
            mu=(1.0, 1.0, 1.0, 1.0),
            lam=3.2,
            alpha=20.0,
            beta=0.85,
            gamma=0.5,
            c=0.5,
        ),
    ]


# ──────────────────────────────────────────────────────────────────────
# Initial-condition bank generation
# ──────────────────────────────────────────────────────────────────────

def generate_initial_conditions(
    N: int,
    *,
    seed: int = 42,
    n_random: int = 10,
) -> list[tuple[str, int, np.ndarray]]:
    """Generate a diverse bank of initial conditions.

    Returns a list of (family_name, index, q0) tuples.

    Families:
        - ``origin``: all queues at zero
        - ``uniform_low``: all queues at 1.0
        - ``uniform_high``: all queues at 50.0
        - ``single_loaded``: one queue at 100, others at zero
        - ``alternating``: alternating high/low queues
        - ``random_uniform``: random uniform in [0, 30]
        - ``random_exponential``: random exponential draws

    Parameters
    ----------
    N : int
        Number of servers.
    seed : int
        Random seed for reproducibility.
    n_random : int
        Number of random initial conditions per random family.

    Returns
    -------
    list of (str, int, np.ndarray)
        Initial-condition bank.
    """
    rng = np.random.default_rng(seed)
    bank: list[tuple[str, int, np.ndarray]] = []

    # Deterministic families
    bank.append(("origin", 0, np.zeros(N, dtype=np.float64)))
    bank.append(("uniform_low", 0, np.ones(N, dtype=np.float64)))
    bank.append(("uniform_high", 0, np.full(N, 50.0, dtype=np.float64)))

    for i in range(N):
        q0 = np.zeros(N, dtype=np.float64)
        q0[i] = 100.0
        bank.append(("single_loaded", i, q0))

    alt = np.zeros(N, dtype=np.float64)
    for i in range(N):
        alt[i] = 30.0 if i % 2 == 0 else 0.5
    bank.append(("alternating", 0, alt))

    # Random families
    for k in range(n_random):
        bank.append(("random_uniform", k, rng.uniform(0.0, 30.0, size=N)))
    for k in range(n_random):
        bank.append(("random_exponential", k, rng.exponential(10.0, size=N)))

    return bank


# ──────────────────────────────────────────────────────────────────────
# Potential function H (from z2/05_projected_gradient_structure.md)
# ──────────────────────────────────────────────────────────────────────

def compute_H(
    q: np.ndarray,
    *,
    mu: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> float:
    """Evaluate the convex potential H(q).

    From z2/05_projected_gradient_structure.md §2:
        H(q) = Σ_i μ_i^{1-β} q_i + (λ/α) log W(q)

    Parameters
    ----------
    q : np.ndarray
        Queue state vector.
    mu, lam, alpha, beta, gamma, c : float
        System parameters.

    Returns
    -------
    float
        Value of H(q).
    """
    # Use log-sum-exp trick for numerical stability with extreme states
    log_w = np.log(np.power(mu, gamma)) - alpha * (q + c) / np.power(mu, beta)
    log_W = np.max(log_w) + np.log(np.sum(np.exp(log_w - np.max(log_w))))
    return float(np.sum(np.power(mu, 1.0 - beta) * q) + (lam / alpha) * log_W)


def compute_equilibrium_residual(
    q: np.ndarray,
    *,
    mu: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> float:
    """Compute the equilibrium residual norm.

    At equilibrium, the complementarity conditions must hold.
    This returns the max violation across all coordinates.

    Parameters
    ----------
    q : np.ndarray
        Queue state vector.
    mu, lam, alpha, beta, gamma, c : float
        System parameters.

    Returns
    -------
    float
        Maximum equilibrium residual.
    """
    w = np.power(mu, gamma) * np.exp(-alpha * (q + c) / np.power(mu, beta))
    W = np.sum(w)
    p = w / W
    lam_p = lam * p

    residuals = np.zeros(len(mu), dtype=np.float64)
    for i in range(len(mu)):
        if q[i] > 1e-10:
            # Active: should have λp_i = μ_i
            residuals[i] = abs(lam_p[i] - mu[i])
        else:
            # Inactive: should have λp_i ≤ μ_i
            residuals[i] = max(0.0, lam_p[i] - mu[i])

    return float(np.max(residuals))


# ──────────────────────────────────────────────────────────────────────
# Reflected-ODE integrator
# ──────────────────────────────────────────────────────────────────────

def integrate_reflected_ode(
    spec: SystemSpec,
    q0: np.ndarray,
    *,
    dt: float = DEFAULT_DT,
    max_time: float = DEFAULT_MAX_TIME,
    H_sample_interval: float = DEFAULT_H_SAMPLE_INTERVAL,
) -> tuple[np.ndarray, list[float]]:
    """Integrate the reflected ODE with Lyapunov function sampling.

    Parameters
    ----------
    spec : SystemSpec
        System specification.
    q0 : np.ndarray
        Initial state.
    dt : float
        Integration time step.
    max_time : float
        Maximum integration time.
    H_sample_interval : float
        Time interval between H(q) samples.

    Returns
    -------
    tuple of (np.ndarray, list of float)
        Terminal state and time-ordered list of H values.
    """
    mu = np.asarray(spec.mu, dtype=np.float64)
    q = q0.copy()
    n_steps = int(max_time / dt)
    sample_every = max(1, int(H_sample_interval / dt))

    H_values: list[float] = []

    for step in range(n_steps):
        if step % sample_every == 0:
            H_val = compute_H(
                q, mu=mu, lam=spec.lam, alpha=spec.alpha,
                beta=spec.beta, gamma=spec.gamma, c=spec.c,
            )
            H_values.append(H_val)

        # Compute routing probabilities (numerically stable)
        log_w = np.log(np.power(mu, spec.gamma)) - spec.alpha * (q + spec.c) / np.power(mu, spec.beta)
        log_w -= np.max(log_w)  # Shift for numerical stability
        w = np.exp(log_w)
        W = np.sum(w)
        p = w / W if W > 0 else np.ones(spec.N, dtype=np.float64) / spec.N

        # Drift + orthant projection
        drift = spec.lam * p - mu
        q = np.maximum(0.0, q + dt * drift)

    # Final H sample
    H_values.append(compute_H(
        q, mu=mu, lam=spec.lam, alpha=spec.alpha,
        beta=spec.beta, gamma=spec.gamma, c=spec.c,
    ))

    return q, H_values


def check_H_monotonicity(H_values: list[float]) -> tuple[float, int]:
    """Check monotonicity of H values (should be non-increasing).

    Parameters
    ----------
    H_values : list of float
        Time-ordered Lyapunov function values.

    Returns
    -------
    tuple of (float, int)
        (fraction of steps where H decreased or stayed flat,
         number of monotonicity violations)
    """
    if len(H_values) < 2:
        return 1.0, 0
    violations = 0
    for i in range(1, len(H_values)):
        if H_values[i] > H_values[i - 1] + 1e-10:
            violations += 1
    monotone_fraction = 1.0 - violations / (len(H_values) - 1)
    return monotone_fraction, violations


# ──────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────

def run_convergence_verification(
    systems: list[SystemSpec],
    output_dir: str | Path,
    *,
    dt: float = DEFAULT_DT,
    max_time: float = DEFAULT_MAX_TIME,
    n_random: int = 10,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Run the multi-start convergence verification experiment.

    Parameters
    ----------
    systems : list of SystemSpec
        Systems to verify.
    output_dir : str or Path
        Output directory for CSV data.
    dt : float
        ODE integration time step.
    max_time : float
        Maximum integration time per trajectory.
    n_random : int
        Number of random ICs per family.
    seed : int
        Random seed.

    Returns
    -------
    tuple of Path
        Paths to (trajectory CSV, summary CSV).
    """
    # ── Per-trajectory CSV ──
    traj_columns = [
        Column("system_id", str, "System identifier"),
        Column("ic_family", str, "Initial-condition family"),
        Column("ic_index", int, "Index within IC family"),
        Column("ic_vector", str, "Initial state (JSON)"),
        Column("terminal_state", str, "Terminal state (JSON)"),
        Column("terminal_residual_norm", float, "Equilibrium residual at terminal state"),
        Column("terminal_H", float, "H(q) at terminal state"),
        Column("initial_H", float, "H(q) at initial state"),
        Column("H_monotone_fraction", float, "Fraction of non-increasing H steps"),
        Column("H_violations", int, "Number of H monotonicity violations"),
        Column("converged", bool, "Whether residual < tolerance"),
    ]

    traj_writer = ExperimentCSVWriter(
        experiment_name="reflected_ode_trajectories",
        output_dir=output_dir,
        columns=traj_columns,
        metadata={
            "hypothesis": "H1, H2",
            "theorem_reference": "z2/06_global_convergence_reflected_ode.md",
            "dt": dt,
            "max_time": max_time,
            "n_random": n_random,
            "seed": seed,
        },
    )

    # ── Summary CSV ──
    summary_columns = [
        Column("system_id", str, "System identifier"),
        Column("N", int, "Number of servers"),
        Column("lambda", float, "Arrival rate"),
        Column("rho", float, "Load factor"),
        Column("num_trajectories", int, "Total trajectories simulated"),
        Column("num_converged", int, "Trajectories meeting convergence tolerance"),
        Column("convergence_rate", float, "Fraction converged"),
        Column("terminal_diameter", float, "Max pairwise distance between terminal states"),
        Column("max_residual_norm", float, "Max equilibrium residual across all trajectories"),
        Column("min_H_monotone_fraction", float, "Min H monotonicity fraction"),
        Column("status", str, "PASS or FAIL"),
    ]

    summary_writer = ExperimentCSVWriter(
        experiment_name="reflected_ode_convergence_summary",
        output_dir=output_dir,
        columns=summary_columns,
        metadata={
            "hypothesis": "H1, H2",
            "theorem_reference": "z2/06_global_convergence_reflected_ode.md",
        },
    )

    for spec in systems:
        log.info("Running convergence test: %s (N=%d, ρ=%.4f)", spec.system_id, spec.N, spec.rho)
        mu = np.asarray(spec.mu, dtype=np.float64)

        ics = generate_initial_conditions(spec.N, seed=seed, n_random=n_random)
        terminal_states: list[np.ndarray] = []
        residuals: list[float] = []
        converged_count = 0
        min_monotone_frac = 1.0

        for family, idx, q0 in ics:
            terminal, H_values = integrate_reflected_ode(
                spec, q0, dt=dt, max_time=max_time,
            )
            terminal_states.append(terminal)

            residual = compute_equilibrium_residual(
                terminal, mu=mu, lam=spec.lam, alpha=spec.alpha,
                beta=spec.beta, gamma=spec.gamma, c=spec.c,
            )
            residuals.append(residual)

            mono_frac, violations = check_H_monotonicity(H_values)
            min_monotone_frac = min(min_monotone_frac, mono_frac)

            converged = residual < CONVERGENCE_TOL
            if converged:
                converged_count += 1

            traj_writer.write_row({
                "system_id": spec.system_id,
                "ic_family": family,
                "ic_index": idx,
                "ic_vector": json.dumps(q0.tolist()),
                "terminal_state": json.dumps(terminal.tolist()),
                "terminal_residual_norm": residual,
                "terminal_H": H_values[-1] if H_values else float("nan"),
                "initial_H": H_values[0] if H_values else float("nan"),
                "H_monotone_fraction": mono_frac,
                "H_violations": violations,
                "converged": converged,
            })

        # Compute terminal pairwise diameter
        if len(terminal_states) > 1:
            max_dist = 0.0
            for i in range(len(terminal_states)):
                for j in range(i + 1, len(terminal_states)):
                    dist = float(np.linalg.norm(terminal_states[i] - terminal_states[j]))
                    max_dist = max(max_dist, dist)
            terminal_diameter = max_dist
        else:
            terminal_diameter = 0.0

        max_residual = max(residuals)
        conv_rate = converged_count / len(ics)

        status = "PASS"
        if conv_rate < 1.0:
            status = "FAIL"
        if terminal_diameter > 1e-3:
            status = "FAIL"

        log.info(
            "  %s: diameter=%.6e, max_residual=%.6e, converged=%d/%d, status=%s",
            spec.system_id, terminal_diameter, max_residual,
            converged_count, len(ics), status,
        )

        summary_writer.write_row({
            "system_id": spec.system_id,
            "N": spec.N,
            "lambda": spec.lam,
            "rho": spec.rho,
            "num_trajectories": len(ics),
            "num_converged": converged_count,
            "convergence_rate": conv_rate,
            "terminal_diameter": terminal_diameter,
            "max_residual_norm": max_residual,
            "min_H_monotone_fraction": min_monotone_frac,
            "status": status,
        })

    traj_path = traj_writer.finalize()
    summary_path = summary_writer.finalize()
    return traj_path, summary_path


def configure_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-start reflected-ODE convergence verification.  "
            "Validates Hypotheses H1 and H2 (Theorem 2)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_TIME)
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    log.info("Starting reflected-ODE convergence verification")
    systems = benchmark_systems()
    traj_path, summary_path = run_convergence_verification(
        systems,
        args.output_dir,
        dt=args.dt,
        max_time=args.max_time,
        n_random=args.n_random,
        seed=args.seed,
    )
    log.info("Trajectory data: %s", traj_path)
    log.info("Summary data: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
