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
      paired_delta_se, etc.

What it does not claim:
    - theorem certification
    - broad dominance beyond the declared benchmark policies and seed blocks

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
from omegaconf import OmegaConf

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
from studies.analysis.common.metrics import (  # noqa: E402
    gini_coefficient,
    sojourn_time_estimate,
    time_averaged_queue_lengths,
)
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


@dataclass(frozen=True)
class BenchmarkSpec:
    mu: tuple[float, ...]
    arrival_rate: float
    num_replications: int
    sim_time: float
    sample_interval: float
    burn_in_fraction: float


def load_benchmark_spec(config_name: str) -> BenchmarkSpec:
    raw_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / f"{config_name}.yaml")
    return BenchmarkSpec(
        mu=tuple(float(x) for x in raw_cfg.system.service_rates),
        arrival_rate=float(raw_cfg.system.arrival_rate),
        num_replications=int(raw_cfg.simulation.num_replications),
        sim_time=float(raw_cfg.simulation.ssa.sim_time),
        sample_interval=float(raw_cfg.simulation.ssa.sample_interval),
        burn_in_fraction=float(raw_cfg.simulation.burn_in_fraction),
    )


def _sample_se(values: Sequence[float]) -> float:
    """Compute standard error of the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _paired_delta_stats(reference: Sequence[float], compared: Sequence[float]) -> tuple[float, float]:
    """Compute the paired mean delta and its standard error."""
    ref = np.asarray(reference, dtype=np.float64)
    other = np.asarray(compared, dtype=np.float64)
    if ref.shape != other.shape:
        raise ValueError("Paired delta requires matching per-replication arrays.")
    deltas = other - ref
    return float(np.mean(deltas)), _sample_se(deltas)


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
    config_name: str = DEFAULT_CONFIG_NAME,
    seed_blocks: Sequence[int] = SEED_BLOCKS,
    num_replications: int | None = None,
    sim_time: float | None = None,
    sample_interval: float | None = None,
    burn_in_fraction: float | None = None,
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
    benchmark = load_benchmark_spec(config_name)
    mu = np.asarray(benchmark.mu, dtype=np.float64)
    num_replications = int(benchmark.num_replications if num_replications is None else num_replications)
    sim_time = float(benchmark.sim_time if sim_time is None else sim_time)
    sample_interval = float(benchmark.sample_interval if sample_interval is None else sample_interval)
    burn_in_fraction = float(benchmark.burn_in_fraction if burn_in_fraction is None else burn_in_fraction)
    suite = build_policy_suite(mu)

    # â”€â”€ Policy results CSV â”€â”€
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

    run_dir, _ = create_run_capsule(output_dir, "independent_seed_rerun")
    attach_run_log_handler(run_dir)
    run_metrics_dir = metrics_dir(run_dir)
    write_run_config(
        run_dir,
        {
            "experiment_name": "independent_seed_rerun",
            "output_dir": str(output_dir),
            "config_name": config_name,
            "seed_blocks": list(seed_blocks),
            "num_replications": num_replications,
            "sim_time": sim_time,
            "sample_interval": sample_interval,
            "burn_in_fraction": burn_in_fraction,
        },
    )

    policy_writer = ExperimentCSVWriter(
        experiment_name="benchmark_rerun_policies",
        output_dir=run_metrics_dir,
        columns=policy_columns,
        metadata={
            "hypothesis": "H5",
            "benchmark_mu": list(benchmark.mu),
            "benchmark_lambda": benchmark.arrival_rate,
            "seed_blocks": list(seed_blocks),
            "config_name": config_name,
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
                arrival_rate=benchmark.arrival_rate,
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

    # â”€â”€ Pairwise comparison CSV â”€â”€
    comp_columns = [
        Column("seed_block", int, "Seed block"),
        Column("policy_a", str, "First policy (reference: Reflected UAS)"),
        Column("policy_b", str, "Compared policy"),
        Column("delta_mean_q", float, "Paired mean_q(B) - mean_q(A)"),
        Column("paired_delta_se", float, "SE of paired delta"),
        Column("reflected_mean_q", float, "Mean q for Reflected UAS"),
        Column("compared_mean_q", float, "Mean q for compared policy"),
        Column("improvement_pct", float, "% improvement of A over B"),
    ]

    comp_writer = ExperimentCSVWriter(
        experiment_name="benchmark_rerun_comparisons",
        output_dir=run_metrics_dir,
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
            delta, paired_delta_se = _paired_delta_stats(
                ref["per_rep_q_totals"],
                other["per_rep_q_totals"],
            )
            improvement = 0.0
            if other["mean_q_total"] > 0:
                improvement = delta / other["mean_q_total"] * 100.0

            comp_writer.write_row({
                "seed_block": seed,
                "policy_a": ref_name,
                "policy_b": pdef.name,
                "delta_mean_q": delta,
                "paired_delta_se": paired_delta_se,
                "reflected_mean_q": ref["mean_q_total"],
                "compared_mean_q": other["mean_q_total"],
                "improvement_pct": improvement,
            })

    comp_path = comp_writer.finalize()
    report_path = metadata_path(run_dir, "benchmark_rerun_summary.md")
    lines = [
        "# Independent Seed Benchmark Rerun Summary",
        "",
        "This report provides focused empirical support for the benchmark anchor.",
        "It does not provide theorem certification.",
        "",
        f"- Seed blocks: {list(seed_blocks)}",
        f"- Config: {config_name}",
        f"- Policies: {[p.name for p in suite]}",
        "",
        "## Pairwise Comparisons",
    ]
    ref_name = "Reflected UAS (default)"
    for seed in iter_progress(
        seed_blocks,
        total=len(seed_blocks),
        desc="benchmark seed blocks",
    ):
        for pdef in suite:
            if pdef.name == ref_name:
                continue
            other = all_results[(seed, pdef.name)]
            ref = all_results[(seed, ref_name)]
            delta, paired_delta_se = _paired_delta_stats(
                ref["per_rep_q_totals"],
                other["per_rep_q_totals"],
            )
            lines.append(
                f"- seed={seed}, {ref_name} vs {pdef.name}: "
                f"delta_mean_q={delta:.6f}, paired_delta_se={paired_delta_se:.6f}, "
                f"reflected_mean_q={ref['mean_q_total']:.6f}, compared_mean_q={other['mean_q_total']:.6f}"
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--num-replications", type=int, default=None)
    parser.add_argument("--sim-time", type=float, default=None)
    parser.add_argument("--sample-interval", type=float, default=None)
    parser.add_argument("--burn-in-fraction", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    log.info("Starting independent-seed benchmark rerun")
    policy_path, comp_path = run_benchmark_rerun(
        args.output_dir,
        config_name=args.config_name,
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


