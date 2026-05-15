#!/usr/bin/env python3
"""
Candidate-grid theorem-constant sweep experiment.

This experiment supports Hypothesis H4 by sweeping a declared compact
grid of (beta, gamma, c) parameter candidates and computing the direct
CTMC theorem constants for each.

What it does:
    1. Sweeps a grid of (beta, gamma, c) candidates.
    2. Computes the theorem constants (epsilon, R) for each candidate
       using the direct CTMC route from z2/10.
    3. Verifies the drift inequality on a fixed sampled state bank.
    4. Classifies each candidate as certified / marginal / non_positive.
    5. Highlights the benchmark-default point in the grid.
    6. Outputs all data as CSV (no figures).

Outputs:
    - Sweep CSV: beta, gamma, c, epsilon, R, classification,
      sampled_pass, max_sampled_residual, etc.

What it does not claim:
    - full theorem promotion from sampled evidence alone
    - certification beyond the declared state-bank protocol

References:
    - z2/10_direct_ctmc_quadratic_proof_attempt.md  (Theorem 3)
    - z2/12_thesis_hypotheses.md  (Hypothesis H4)
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from omegaconf import OmegaConf

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
DEFAULT_CONFIG_NAME = "final_experiment"

BENCHMARK_ALPHA = 20.0

BENCHMARK_BETA = 0.85
BENCHMARK_GAMMA = 0.5
BENCHMARK_C = 0.5

DEFAULT_BETA_VALUES = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5]
DEFAULT_GAMMA_VALUES = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
DEFAULT_C_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

SAMPLED_BANK_SEED = 20260513
SAMPLED_BANK_PER_SHELL = 64


def load_benchmark_system(config_name: str) -> tuple[tuple[float, ...], float]:
    raw_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / f"{config_name}.yaml")
    mu = tuple(float(x) for x in raw_cfg.system.service_rates)
    arrival_rate = float(raw_cfg.system.arrival_rate)
    return mu, arrival_rate


def _axis_states(num_servers: int, magnitudes: Sequence[int]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for i in range(num_servers):
        for mag in magnitudes:
            state = np.zeros(num_servers, dtype=np.int64)
            state[i] = int(mag)
            rows.append(state)
    return np.asarray(rows, dtype=np.int64)


def _paired_imbalance_states(num_servers: int, magnitudes: Sequence[int]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for i in range(num_servers):
        for j in range(num_servers):
            if i == j:
                continue
            for mag in magnitudes:
                state = np.zeros(num_servers, dtype=np.int64)
                state[i] = int(mag)
                state[j] = int(mag // 4)
                rows.append(state)
    return np.asarray(rows, dtype=np.int64)


def _shell_random_states(
    *,
    num_servers: int,
    norms: Sequence[int],
    per_shell: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    probs = np.full(num_servers, 1.0 / num_servers, dtype=np.float64)
    for norm in norms:
        for _ in range(per_shell):
            rows.append(rng.multinomial(int(norm), probs))
    return np.asarray(rows, dtype=np.int64)


def build_sampled_state_bank(num_servers: int) -> np.ndarray:
    """Build the fixed sampled state bank used for sweep classification."""
    chunks = [
        np.zeros((1, num_servers), dtype=np.int64),
        _axis_states(num_servers, magnitudes=(1, 2, 5, 10, 20, 40, 80, 160)),
        _paired_imbalance_states(num_servers, magnitudes=(4, 8, 16, 32, 64, 128)),
        _shell_random_states(
            num_servers=num_servers,
            norms=(5, 10, 20, 40, 80, 160),
            per_shell=SAMPLED_BANK_PER_SHELL,
            seed=SAMPLED_BANK_SEED,
        ),
    ]
    return np.unique(np.vstack(chunks), axis=0)


def compute_theorem_constants(
    *,
    mu: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> dict[str, float]:
    """Compute the direct CTMC theorem constants."""
    mu_beta = np.power(mu, beta)
    service_term = np.power(mu, 1.0 - beta)
    kappa = c / mu_beta - (gamma / alpha) * np.log(mu)
    total_capacity = float(np.sum(mu))

    c1 = float(math.log(len(mu)) / alpha + np.max(kappa) - np.min(kappa))
    c0 = float(0.5 * lam * np.max(np.power(mu, -beta)) + 0.5 * np.sum(service_term))
    r_value = lam * c1 + c0
    epsilon = min(
        (total_capacity - lam) / float(np.sum(mu_beta)),
        float(np.min(service_term)),
    )

    return {
        "epsilon": float(epsilon),
        "R": float(r_value),
        "C0": float(c0),
        "C1": float(c1),
        "load_margin": float(total_capacity - lam),
        "sum_mu_beta": float(np.sum(mu_beta)),
        "min_service_term": float(np.min(service_term)),
    }


def exact_quadratic_generator_drift(
    states: np.ndarray,
    *,
    mu: np.ndarray,
    lam: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> np.ndarray:
    """Compute the exact weighted-quadratic generator drift on a state bank."""
    q = np.asarray(states, dtype=np.int64)
    if q.ndim != 2:
        raise ValueError("states must have shape (M, N)")

    mu_beta = np.power(mu, beta)
    log_w = gamma * np.log(mu)[None, :] - alpha * (q.astype(np.float64) + c) / mu_beta[None, :]
    log_w -= np.max(log_w, axis=1, keepdims=True)
    weights = np.exp(log_w)
    probs = weights / np.sum(weights, axis=1, keepdims=True)

    q_float = q.astype(np.float64)
    arrival = lam * np.sum(probs * ((q_float + 0.5) / mu_beta[None, :]), axis=1)
    departure = np.sum(
        mu[None, :] * (q > 0).astype(np.float64) * ((q_float - 0.5) / mu_beta[None, :]),
        axis=1,
    )
    return arrival - departure


def classify_candidate(epsilon: float, R: float) -> str:
    """Classify a candidate from constants alone."""
    if epsilon <= 0:
        return "non_positive"
    if epsilon > 1e-10:
        return "certified"
    return "marginal"


def run_theorem_constant_sweep(
    output_dir: str | Path,
    *,
    config_name: str = DEFAULT_CONFIG_NAME,
    beta_values: Sequence[float] = DEFAULT_BETA_VALUES,
    gamma_values: Sequence[float] = DEFAULT_GAMMA_VALUES,
    c_values: Sequence[float] = DEFAULT_C_VALUES,
) -> Path:
    """Run the theorem-constant sweep."""
    benchmark_mu, benchmark_lambda = load_benchmark_system(config_name)
    mu = np.asarray(benchmark_mu, dtype=np.float64)
    sampled_states = build_sampled_state_bank(len(mu))

    columns = [
        Column("beta", float, "Service-rate exponent beta"),
        Column("gamma", float, "Prefactor exponent gamma"),
        Column("c", float, "Queue-offset constant c"),
        Column("alpha", float, "Softmax inverse-temperature alpha"),
        Column("epsilon", float, "Theorem drift-rate constant epsilon"),
        Column("R", float, "Theorem remainder constant R"),
        Column("C0", float, "Constant C0"),
        Column("C1", float, "Constant C1"),
        Column("load_margin", float, "Lambda - lambda"),
        Column("sum_mu_beta", float, "Sum mu_i^beta"),
        Column("min_service_term", float, "min(mu_i^(1-beta))"),
        Column("classification", str, "certified / marginal / non_positive"),
        Column("is_benchmark_default", bool, "Whether this is the benchmark point"),
        Column("compact_set_radius", float, "R/epsilon"),
        Column("sampled_state_bank_size", int, "Number of sampled states used"),
        Column("sampled_pass", bool, "Whether the sampled drift audit passed"),
        Column("positive_violation_count", int, "Sampled states with positive residual"),
        Column("max_sampled_residual", float, "Max sampled residual"),
    ]

    total_candidates = len(beta_values) * len(gamma_values) * len(c_values)
    log.info(
        "Sweeping %d candidates (%d beta x %d gamma x %d c)",
        total_candidates,
        len(beta_values),
        len(gamma_values),
        len(c_values),
    )

    run_dir, _ = create_run_capsule(output_dir, "theorem_constant_sweep")
    attach_run_log_handler(run_dir)
    write_run_config(
        run_dir,
        {
            "experiment_name": "theorem_constant_sweep",
            "output_dir": str(output_dir),
            "config_name": config_name,
            "beta_values": list(beta_values),
            "gamma_values": list(gamma_values),
            "c_values": list(c_values),
        },
    )

    writer = ExperimentCSVWriter(
        experiment_name="theorem_constant_sweep",
        output_dir=metrics_dir(run_dir),
        columns=columns,
        metadata={
            "hypothesis": "H4",
            "theorem_reference": "z2/10_direct_ctmc_quadratic_proof_attempt.md",
            "benchmark_mu": list(benchmark_mu),
            "benchmark_lambda": benchmark_lambda,
            "benchmark_alpha": BENCHMARK_ALPHA,
            "config_name": config_name,
            "beta_values": list(beta_values),
            "gamma_values": list(gamma_values),
            "c_values": list(c_values),
            "total_candidates": total_candidates,
            "sampled_state_bank_size": int(sampled_states.shape[0]),
            "sampled_state_bank_seed": SAMPLED_BANK_SEED,
        },
    )

    n_certified = 0
    n_non_positive = 0

    parameter_grid = list(itertools.product(beta_values, gamma_values, c_values))
    for beta, gamma, c_val in iter_progress(
        parameter_grid,
        total=len(parameter_grid),
        desc="theorem sweep",
    ):
        if beta <= 0:
            log.warning("Skipping invalid beta=%.4f", beta)
            continue

        constants = compute_theorem_constants(
            mu=mu,
            lam=benchmark_lambda,
            alpha=BENCHMARK_ALPHA,
            beta=beta,
            gamma=gamma,
            c=c_val,
        )

        drifts = exact_quadratic_generator_drift(
            sampled_states,
            mu=mu,
            lam=benchmark_lambda,
            alpha=BENCHMARK_ALPHA,
            beta=beta,
            gamma=gamma,
            c=c_val,
        )
        norms = sampled_states.sum(axis=1).astype(np.float64)
        theorem_rhs = -constants["epsilon"] * norms + constants["R"]
        residuals = drifts - theorem_rhs
        positive_violation_count = int(np.count_nonzero(residuals > 1e-9))
        sampled_pass = bool(positive_violation_count == 0)
        max_sampled_residual = float(np.max(residuals))

        if constants["epsilon"] <= 0.0:
            classification = "non_positive"
        elif sampled_pass:
            classification = "certified"
        else:
            classification = "marginal"

        is_default = (
            abs(beta - BENCHMARK_BETA) < 1e-10
            and abs(gamma - BENCHMARK_GAMMA) < 1e-10
            and abs(c_val - BENCHMARK_C) < 1e-10
        )
        compact_radius = (
            constants["R"] / constants["epsilon"]
            if constants["epsilon"] > 0
            else float("inf")
        )

        if classification == "certified":
            n_certified += 1
        elif classification == "non_positive":
            n_non_positive += 1

        writer.write_row(
            {
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
                "sampled_state_bank_size": int(sampled_states.shape[0]),
                "sampled_pass": sampled_pass,
                "positive_violation_count": positive_violation_count,
                "max_sampled_residual": max_sampled_residual,
            }
        )

    log.info(
        "Sweep complete: %d certified, %d non-positive, %d total",
        n_certified,
        n_non_positive,
        total_candidates,
    )

    csv_path = writer.finalize()
    report_path = metadata_path(run_dir, "theorem_constant_sweep_summary.md")
    lines = [
        "# Theorem Constant Sweep Summary",
        "",
        "This report classifies candidates by positive epsilon and sampled",
        "drift-bound support. It does not promote sampled evidence into a final theorem.",
        "",
        f"- Total candidates: {total_candidates}",
        f"- Certified: {n_certified}",
        f"- Non-positive: {n_non_positive}",
        f"- Marginal: {total_candidates - n_certified - n_non_positive}",
        "",
        f"- Benchmark default point: beta={BENCHMARK_BETA}, gamma={BENCHMARK_GAMMA}, c={BENCHMARK_C}",
        f"- Sampled state bank size: {sampled_states.shape[0]}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Theorem-constant sweep over (beta, gamma, c) grid (H4 support).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    log.info("Starting theorem-constant sweep")
    csv_path = run_theorem_constant_sweep(args.output_dir, config_name=args.config_name)
    log.info("Sweep data: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
