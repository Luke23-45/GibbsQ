#!/usr/bin/env python3
"""
Independent-seed benchmark rerun experiment.

This experiment supports Hypothesis H5 by running policy comparisons
on a fresh seed block that is independent of the original anchor
benchmark, reducing the risk of seed-specific artifacts.

What it does:
    1. Runs UAS, Reflected UAS (benchmark default), and JSSQ under
       the declared benchmark protocol with independent seeds.
    2. Computes mean total queue, standard error, Gini coefficient,
       and sojourn time.
    3. Computes pairwise deltas relative to Reflected UAS.
    4. Outputs all data as CSV (no figures).

Outputs:
    - Per-policy CSV: policy, seed_block, mean_q_total, se_q_total,
      mean_gini, se_gini, mean_sojourn, se_sojourn, etc.
    - Pairwise comparison CSV: policy_a, policy_b, delta_mean_q,
      delta_se, etc.

References:
    - z2/12_thesis_hypotheses.md  (Hypothesis H5)
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gibbsq.qroute.core.policies import (  # noqa: E402
    ReflectedUASRouting,
    JSSQRouting,
    UASRouting,
)
from gibbsq.qroute.engines.numpy_engine import run_replications  # noqa: E402
from gibbsq.qroute.analysis.metrics import (  # noqa: E402
    gini_coefficient,
    sojourn_time_estimate,
    time_averaged_queue_lengths,
)
from gibbsq.qroute.utils.csv_writer import Column, ExperimentCSVWriter  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "outputs/data"

# Benchmark protocol
BENCHMARK_MU = tuple(0.5 + 0.2 * i for i in range(10))
BENCHMARK_LAMBDA = 11.2
BENCHMARK_NUM_SERVERS = 10

# Independent seed blocks (non-overlapping with anchor seed=42)
SEED_BLOCKS = [1000, 2000, 3000]

DEFAULT_NUM_REPLICATIONS = 32
DEFAULT_SIM_TIME = 15000.0
DEFAULT_SAMPLE_INTERVAL = 1.0
DEFAULT_BURN_IN_FRACTION = 0.2


@dataclass(frozen=True)
class PolicyDef:
    """Policy definition for benchmark comparison."""

    name: str
    family: str
    policy: object  # routing policy object

    def __repr__(self) -> str:
        return f"PolicyDef(name={self.name!r}, family={self.family!r})"


def _sample_se(values: Sequence[float]) -> float:
    """Compute standard error of the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def build_policy_suite(mu: np.ndarray) -> list[PolicyDef]:
    """Build the benchmark policy suite.

    Parameters
    ----------
    mu : np.ndarray
        Service rates.

    Returns
    -------
    list of PolicyDef
        Ordered policy definitions.
    """
    return [
        PolicyDef(
            name="UAS",
            family="uas_baseline",
            policy=UASRouting(mu=mu, alpha=10.0),
        ),
        PolicyDef(
            name="Reflected UAS (default)",
            family="reflected_uas",
            policy=ReflectedUASRouting(
                mu=mu, alpha=20.0, beta=0.85, gamma=0.5, c=0.5,
            ),
        ),
        PolicyDef(
            name="JSSQ",
            family="jssq_baseline",
            policy=JSSQRouting(mu=mu),
        ),
    ]


def evaluate_policy(
    *,
    policy_def: PolicyDef,
    mu: np.ndarray,
    arrival_rate: float,
    num_replications: int,
    sim_time: float,
    sample_interval: float,
    burn_in_fraction: float,
    base_seed: int,
) -> dict[str, object]:
    """Run simulation replications for a single policy.

    Parameters
    ----------
    policy_def : PolicyDef
        Policy to evaluate.
    mu : np.ndarray
        Service rates.
    arrival_rate : float
        Arrival rate.
    num_replications : int
        Number of independent replications.
    sim_time : float
        Simulation horizon per replication.
    sample_interval : float
        State sampling interval.
    burn_in_fraction : float
        Fraction of initial samples to discard.
    base_seed : int
        Base random seed.

    Returns
    -------
    dict
        Aggregated metrics.
    """
    results = run_replications(
        num_servers=len(mu),
        arrival_rate=arrival_rate,
        service_rates=mu,
        policy=policy_def.policy,
        num_replications=num_replications,
        sim_time=sim_time,
        sample_interval=sample_interval,
        base_seed=base_seed,
        progress_desc=f"rerun ({policy_def.name}, seed={base_seed})",
    )

    q_totals = [
        float(time_averaged_queue_lengths(r, burn_in_fraction).sum())
        for r in results
    ]
    ginis = [
        float(gini_coefficient(time_averaged_queue_lengths(r, burn_in_fraction)))
        for r in results
    ]
    sojourns = [
        float(sojourn_time_estimate(r, arrival_rate, burn_in_fraction))
        for r in results
    ]

    return {
        "mean_q_total": float(np.mean(q_totals)),
        "se_q_total": _sample_se(q_totals),
        "mean_gini": float(np.mean(ginis)),
        "se_gini": _sample_se(ginis),
        "mean_sojourn": float(np.mean(sojourns)),
        "se_sojourn": _sample_se(sojourns),
        "per_rep_q_totals": q_totals,
    }


def run_benchmark_rerun(
    output_dir: str | Path,
    *,
    seed_blocks: Sequence[int] = SEED_BLOCKS,
    num_replications: int = DEFAULT_NUM_REPLICATIONS,
    sim_time: float = DEFAULT_SIM_TIME,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    burn_in_fraction: float = DEFAULT_BURN_IN_FRACTION,
) -> tuple[Path, Path]:
    """Run the independent-seed benchmark rerun.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.
    seed_blocks : sequence of int
        Independent seed values.
    num_replications : int
        Replications per seed block.
    sim_time : float
        Simulation time per replication.
    sample_interval : float
        State sampling interval.
    burn_in_fraction : float
        Burn-in fraction.

    Returns
    -------
    tuple of Path
        Paths to (policy CSV, comparison CSV).
    """
    mu = np.asarray(BENCHMARK_MU, dtype=np.float64)
    suite = build_policy_suite(mu)

    # ── Policy results CSV ──
    policy_columns = [
        Column("policy", str, "Policy name"),
        Column("family", str, "Policy family"),
        Column("seed_block", int, "Base random seed"),
        Column("num_replications", int, "Number of replications"),
        Column("sim_time", float, "Simulation time per replication"),
        Column("mean_q_total", float, "Mean total queue length"),
        Column("se_q_total", float, "Standard error of mean total queue"),
        Column("mean_gini", float, "Mean Gini coefficient"),
        Column("se_gini", float, "SE of Gini coefficient"),
        Column("mean_sojourn", float, "Mean sojourn time"),
        Column("se_sojourn", float, "SE of sojourn time"),
        Column("per_rep_q_totals", str, "Per-replication total queues (JSON)"),
    ]

    policy_writer = ExperimentCSVWriter(
        experiment_name="benchmark_rerun_policies",
        output_dir=output_dir,
        columns=policy_columns,
        metadata={
            "hypothesis": "H5",
            "benchmark_mu": list(BENCHMARK_MU),
            "benchmark_lambda": BENCHMARK_LAMBDA,
            "seed_blocks": list(seed_blocks),
        },
    )

    # Collect results keyed by (seed_block, policy_name)
    all_results: dict[tuple[int, str], dict[str, object]] = {}

    for seed in seed_blocks:
        log.info("Seed block: %d", seed)
        for pdef in suite:
            log.info("  Evaluating: %s", pdef.name)
            metrics = evaluate_policy(
                policy_def=pdef,
                mu=mu,
                arrival_rate=BENCHMARK_LAMBDA,
                num_replications=num_replications,
                sim_time=sim_time,
                sample_interval=sample_interval,
                burn_in_fraction=burn_in_fraction,
                base_seed=seed,
            )

            policy_writer.write_row({
                "policy": pdef.name,
                "family": pdef.family,
                "seed_block": seed,
                "num_replications": num_replications,
                "sim_time": sim_time,
                "mean_q_total": metrics["mean_q_total"],
                "se_q_total": metrics["se_q_total"],
                "mean_gini": metrics["mean_gini"],
                "se_gini": metrics["se_gini"],
                "mean_sojourn": metrics["mean_sojourn"],
                "se_sojourn": metrics["se_sojourn"],
                "per_rep_q_totals": json.dumps(metrics["per_rep_q_totals"]),
            })

            all_results[(seed, pdef.name)] = metrics

    policy_path = policy_writer.finalize()

    # ── Pairwise comparison CSV ──
    comp_columns = [
        Column("seed_block", int, "Seed block"),
        Column("policy_a", str, "First policy (reference: Reflected UAS)"),
        Column("policy_b", str, "Compared policy"),
        Column("delta_mean_q", float, "mean_q(B) - mean_q(A)"),
        Column("reflected_mean_q", float, "Mean q for Reflected UAS"),
        Column("compared_mean_q", float, "Mean q for compared policy"),
        Column("improvement_pct", float, "% improvement of A over B"),
    ]

    comp_writer = ExperimentCSVWriter(
        experiment_name="benchmark_rerun_comparisons",
        output_dir=output_dir,
        columns=comp_columns,
        metadata={"hypothesis": "H5"},
    )

    ref_name = "Reflected UAS (default)"
    for seed in seed_blocks:
        ref = all_results[(seed, ref_name)]
        for pdef in suite:
            if pdef.name == ref_name:
                continue
            other = all_results[(seed, pdef.name)]
            delta = other["mean_q_total"] - ref["mean_q_total"]
            improvement = 0.0
            if other["mean_q_total"] > 0:
                improvement = delta / other["mean_q_total"] * 100.0

            comp_writer.write_row({
                "seed_block": seed,
                "policy_a": ref_name,
                "policy_b": pdef.name,
                "delta_mean_q": delta,
                "reflected_mean_q": ref["mean_q_total"],
                "compared_mean_q": other["mean_q_total"],
                "improvement_pct": improvement,
            })

    comp_path = comp_writer.finalize()
    return policy_path, comp_path


def configure_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Independent-seed benchmark rerun (H5 support).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-replications", type=int, default=DEFAULT_NUM_REPLICATIONS)
    parser.add_argument("--sim-time", type=float, default=DEFAULT_SIM_TIME)
    parser.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--burn-in-fraction", type=float, default=DEFAULT_BURN_IN_FRACTION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    log.info("Starting independent-seed benchmark rerun")
    policy_path, comp_path = run_benchmark_rerun(
        args.output_dir,
        num_replications=args.num_replications,
        sim_time=args.sim_time,
        sample_interval=args.sample_interval,
        burn_in_fraction=args.burn_in_fraction,
    )
    log.info("Policy results: %s", policy_path)
    log.info("Comparison results: %s", comp_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
