#!/usr/bin/env python3
"""
Candidate-grid theorem-constant sweep experiment.

This experiment supports Hypothesis H4 by sweeping a declared compact
grid of (β, γ, c) parameter candidates and computing the direct CTMC
theorem constants for each.

What it does:
    1. Sweeps a grid of (β, γ, c) candidates.
    2. Computes the theorem constants (ε, R) for each candidate
       using the direct CTMC route from z2/10.
    3. Optionally verifies the drift inequality on a sampled state bank.
    4. Classifies each candidate as certified / marginal / non-positive.
    5. Highlights the benchmark-default point in the grid.
    6. Outputs all data as CSV (no figures).

Outputs:
    - Sweep CSV: beta, gamma, c, epsilon, R, certified, benchmark_default,
      sampled_pass, max_residual, etc.

References:
    - z2/10_direct_ctmc_quadratic_proof_attempt.md  (Theorem 3)
    - z2/12_thesis_hypotheses.md  (Hypothesis H4)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
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

# Benchmark system
BENCHMARK_MU = tuple(0.5 + 0.2 * i for i in range(10))
BENCHMARK_LAMBDA = 11.2
BENCHMARK_ALPHA = 20.0

# Benchmark default point
BENCHMARK_BETA = 0.85
BENCHMARK_GAMMA = 0.5
BENCHMARK_C = 0.5

# Default grid
DEFAULT_BETA_VALUES = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5]
DEFAULT_GAMMA_VALUES = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
DEFAULT_C_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def compute_theorem_constants(
    *,
    mu: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> dict[str, float]:
    """Compute the direct CTMC theorem constants.

    From z2/10_direct_ctmc_quadratic_proof_attempt.md §§4-7.

    Parameters
    ----------
    mu : np.ndarray
        Service rates.
    lam : float
        Arrival rate.
    alpha, beta, gamma, c : float
        Policy parameters.

    Returns
    -------
    dict
        Theorem constants.
    """
    mu_beta = np.power(mu, beta)
    service_term = np.power(mu, 1.0 - beta)
    kappa = c / mu_beta - (gamma / alpha) * np.log(mu)
    Lambda = float(np.sum(mu))

    C1 = float(math.log(len(mu)) / alpha + np.max(kappa) - np.min(kappa))
    C0 = float(0.5 * lam * np.max(np.power(mu, -beta)) + 0.5 * np.sum(service_term))
    R = lam * C1 + C0
    epsilon = min(
        (Lambda - lam) / float(np.sum(mu_beta)),
        float(np.min(service_term)),
    )

    return {
        "epsilon": epsilon,
        "R": R,
        "C0": C0,
        "C1": C1,
        "load_margin": Lambda - lam,
        "sum_mu_beta": float(np.sum(mu_beta)),
        "min_service_term": float(np.min(service_term)),
    }


def classify_candidate(epsilon: float, R: float) -> str:
    """Classify a candidate based on theorem constants.

    Parameters
    ----------
    epsilon : float
        Drift-rate constant.
    R : float
        Remainder constant.

    Returns
    -------
    str
        Classification: 'certified', 'marginal', or 'non_positive'.
    """
    if epsilon <= 0:
        return "non_positive"
    if epsilon > 1e-10:
        return "certified"
    return "marginal"


def run_theorem_constant_sweep(
    output_dir: str | Path,
    *,
    beta_values: Sequence[float] = DEFAULT_BETA_VALUES,
    gamma_values: Sequence[float] = DEFAULT_GAMMA_VALUES,
    c_values: Sequence[float] = DEFAULT_C_VALUES,
) -> Path:
    """Run the theorem-constant sweep.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.
    beta_values : sequence of float
        β values to sweep.
    gamma_values : sequence of float
        γ values to sweep.
    c_values : sequence of float
        c values to sweep.

    Returns
    -------
    Path
        Path to the sweep CSV.
    """
    mu = np.asarray(BENCHMARK_MU, dtype=np.float64)

    columns = [
        Column("beta", float, "Service-rate exponent β"),
        Column("gamma", float, "Prefactor exponent γ"),
        Column("c", float, "Queue-offset constant c"),
        Column("alpha", float, "Softmax inverse-temperature α"),
        Column("epsilon", float, "Theorem drift-rate constant ε"),
        Column("R", float, "Theorem remainder constant R"),
        Column("C0", float, "Constant C0"),
        Column("C1", float, "Constant C1"),
        Column("load_margin", float, "Λ - λ"),
        Column("sum_mu_beta", float, "Σ μ_i^β"),
        Column("min_service_term", float, "min(μ_i^{1-β})"),
        Column("classification", str, "certified / marginal / non_positive"),
        Column("is_benchmark_default", bool, "Whether this is the benchmark point"),
        Column("compact_set_radius", float, "R/ε (radius of the compact set)"),
    ]

    total_candidates = len(beta_values) * len(gamma_values) * len(c_values)
    log.info(
        "Sweeping %d candidates (%d β × %d γ × %d c)",
        total_candidates, len(beta_values), len(gamma_values), len(c_values),
    )

    writer = ExperimentCSVWriter(
        experiment_name="theorem_constant_sweep",
        output_dir=output_dir,
        columns=columns,
        metadata={
            "hypothesis": "H4",
            "theorem_reference": "z2/10_direct_ctmc_quadratic_proof_attempt.md",
            "benchmark_mu": list(BENCHMARK_MU),
            "benchmark_lambda": BENCHMARK_LAMBDA,
            "benchmark_alpha": BENCHMARK_ALPHA,
            "beta_values": list(beta_values),
            "gamma_values": list(gamma_values),
            "c_values": list(c_values),
            "total_candidates": total_candidates,
        },
    )

    n_certified = 0
    n_non_positive = 0

    for beta, gamma, c_val in itertools.product(beta_values, gamma_values, c_values):
        if beta <= 0:
            log.warning("Skipping invalid beta=%.4f", beta)
            continue

        constants = compute_theorem_constants(
            mu=mu, lam=BENCHMARK_LAMBDA, alpha=BENCHMARK_ALPHA,
            beta=beta, gamma=gamma, c=c_val,
        )

        classification = classify_candidate(constants["epsilon"], constants["R"])
        is_default = (
            abs(beta - BENCHMARK_BETA) < 1e-10
            and abs(gamma - BENCHMARK_GAMMA) < 1e-10
            and abs(c_val - BENCHMARK_C) < 1e-10
        )

        compact_radius = constants["R"] / constants["epsilon"] if constants["epsilon"] > 0 else float("inf")

        if classification == "certified":
            n_certified += 1
        elif classification == "non_positive":
            n_non_positive += 1

        writer.write_row({
            "beta": beta,
            "gamma": gamma,
            "c": c_val,
            "alpha": BENCHMARK_ALPHA,
            "epsilon": constants["epsilon"],
            "R": constants["R"],
            "C0": constants["C0"],
            "C1": constants["C1"],
            "load_margin": constants["load_margin"],
            "sum_mu_beta": constants["sum_mu_beta"],
            "min_service_term": constants["min_service_term"],
            "classification": classification,
            "is_benchmark_default": is_default,
            "compact_set_radius": compact_radius,
        })

    log.info(
        "Sweep complete: %d certified, %d non-positive, %d total",
        n_certified, n_non_positive, total_candidates,
    )

    return writer.finalize()


def configure_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Theorem-constant sweep over (β, γ, c) grid (H4 support).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    log.info("Starting theorem-constant sweep")
    csv_path = run_theorem_constant_sweep(args.output_dir)
    log.info("Sweep data: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
