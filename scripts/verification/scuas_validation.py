#!/usr/bin/env python3
"""
SCUAS validation runner for theorem certification and calibrated-policy reruns.

This script provides a dedicated workflow for the manuscript update that
introduces a theorem-backed Stability-Certified Calibrated UAS (SCUAS)
subfamily without overclaiming the full empirical Calibrated UAS family.

Modes
-----
`audit`
    Compute the SCUAS theorem constants on the active benchmark setup.
`rerun`
    Rerun the anchor policy comparison for UAS, empirical Calibrated UAS,
    and only those SCUAS candidates with epsilon > 0.
`full`
    Run both audit and rerun in one capsule.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gibbsq.analysis.metrics import (  # noqa: E402
    gini_coefficient,
    sojourn_time_estimate,
    time_averaged_queue_lengths,
)
from gibbsq.core.config import (  # noqa: E402
    ExperimentConfig,
    _profile_path,
    hydra_to_config,
    load_experiment_config,
    validate,
)
from gibbsq.core.policies import CalibratedUASRouting, UASRouting  # noqa: E402
from gibbsq.engines.numpy_engine import run_replications  # noqa: E402
from gibbsq.utils.exporter import append_metrics_jsonl  # noqa: E402
from gibbsq.utils.logging import get_run_config  # noqa: E402
from gibbsq.utils.run_artifacts import metadata_path, metrics_path  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "final_experiment"
DEFAULT_MODE = "full"
ANCHOR_REPLICATIONS = 32
ANCHOR_SIM_TIME = 15000.0
ANCHOR_SAMPLE_INTERVAL = 1.0
ANCHOR_BURN_IN_FRACTION = 0.2
ANCHOR_BASE_SEED = 42
DEFAULT_UAS_ALPHA = 10.0
DEFAULT_CALIBRATED_ALPHA = 20.0


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    beta: float
    gamma: float
    c: float
    family: str
    notes: str = ""

    def triplet(self) -> tuple[float, float, float]:
        return (self.beta, self.gamma, self.c)


@dataclass(frozen=True)
class ValidationProtocol:
    num_replications: int = ANCHOR_REPLICATIONS
    sim_time: float = ANCHOR_SIM_TIME
    sample_interval: float = ANCHOR_SAMPLE_INTERVAL
    burn_in_fraction: float = ANCHOR_BURN_IN_FRACTION
    base_seed: int = ANCHOR_BASE_SEED
    alpha_uas: float = DEFAULT_UAS_ALPHA
    alpha_calibrated: float = DEFAULT_CALIBRATED_ALPHA


def default_candidate_catalog() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="calibrated_default",
            beta=0.85,
            gamma=0.5,
            c=0.5,
            family="empirical_calibrated_default",
            notes="Current manuscript default empirical calibrated policy.",
        ),
        CandidateSpec(
            name="scuas_candidate_b0p5_g0p25_c0p25",
            beta=0.5,
            gamma=0.25,
            c=0.25,
            family="scuas_candidate",
            notes="Grid candidate proposed for theorem audit.",
        ),
        CandidateSpec(
            name="scuas_candidate_b0p5_g0p5_c0p25",
            beta=0.5,
            gamma=0.5,
            c=0.25,
            family="scuas_candidate",
            notes="Grid candidate proposed for theorem audit.",
        ),
        CandidateSpec(
            name="scuas_candidate_b0p7_g0p25_c0p25",
            beta=0.7,
            gamma=0.25,
            c=0.25,
            family="scuas_candidate",
            notes="Grid candidate proposed for theorem audit.",
        ),
    ]


def parse_candidate_triplet(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Candidate '{text}' must have exactly three comma-separated values: beta,gamma,c."
        )
    try:
        beta, gamma, c = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Candidate '{text}' contains a non-numeric value.") from exc
    _validate_candidate_values(beta=beta, gamma=gamma, c=c)
    return beta, gamma, c


def make_candidate_name(beta: float, gamma: float, c: float) -> str:
    def _fmt(value: float) -> str:
        return str(value).replace("-", "m").replace(".", "p")

    return f"candidate_b{_fmt(beta)}_g{_fmt(gamma)}_c{_fmt(c)}"


def _validate_candidate_values(*, beta: float, gamma: float, c: float) -> None:
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if c < 0:
        raise ValueError(f"c must be >= 0, got {c}")
    if not math.isfinite(beta) or not math.isfinite(gamma) or not math.isfinite(c):
        raise ValueError("Candidate parameters must be finite real values.")


def extend_candidate_catalog(extra_triplets: Sequence[str]) -> list[CandidateSpec]:
    catalog = list(default_candidate_catalog())
    known = {candidate.triplet() for candidate in catalog}
    for raw_triplet in extra_triplets:
        beta, gamma, c = parse_candidate_triplet(raw_triplet)
        triplet = (beta, gamma, c)
        if triplet in known:
            continue
        catalog.append(
            CandidateSpec(
                name=make_candidate_name(beta, gamma, c),
                beta=beta,
                gamma=gamma,
                c=c,
                family="scuas_candidate",
                notes="User-specified SCUAS candidate.",
            )
        )
        known.add(triplet)
    return catalog


def load_policy_experiment_config(
    *,
    config_name: str,
    overrides: Sequence[str],
    protocol: ValidationProtocol,
    output_dir: str | None,
) -> tuple[ExperimentConfig, DictConfig]:
    raw_profile = OmegaConf.load(_profile_path(config_name))
    if overrides:
        raw_profile = OmegaConf.merge(raw_profile, OmegaConf.from_dotlist(list(overrides)))

    cfg, resolved_raw = load_experiment_config(raw_profile, "policy", profile_name=config_name)

    OmegaConf.update(resolved_raw, "simulation.num_replications", int(protocol.num_replications))
    OmegaConf.update(resolved_raw, "simulation.ssa.sim_time", float(protocol.sim_time))
    OmegaConf.update(resolved_raw, "simulation.ssa.sample_interval", float(protocol.sample_interval))
    OmegaConf.update(resolved_raw, "simulation.burn_in_fraction", float(protocol.burn_in_fraction))
    OmegaConf.update(resolved_raw, "simulation.seed", int(protocol.base_seed))
    OmegaConf.update(resolved_raw, "wandb.enabled", False)
    if output_dir is not None:
        OmegaConf.update(resolved_raw, "output_dir", output_dir)

    cfg = hydra_to_config(resolved_raw)
    validate(cfg)
    _validate_protocol(cfg, protocol)
    return cfg, resolved_raw


def _validate_protocol(cfg: ExperimentConfig, protocol: ValidationProtocol) -> None:
    if cfg.system.num_servers != len(cfg.system.service_rates):
        raise ValueError(
            "Resolved config has inconsistent num_servers and service_rates lengths after overrides."
        )
    if protocol.num_replications < 1:
        raise ValueError(f"num_replications must be >= 1, got {protocol.num_replications}")
    if protocol.sim_time <= 0:
        raise ValueError(f"sim_time must be > 0, got {protocol.sim_time}")
    if protocol.sample_interval <= 0:
        raise ValueError(f"sample_interval must be > 0, got {protocol.sample_interval}")
    if not (0.0 <= protocol.burn_in_fraction < 1.0):
        raise ValueError(
            f"burn_in_fraction must lie in [0, 1), got {protocol.burn_in_fraction}"
        )
    if protocol.alpha_uas <= 0 or protocol.alpha_calibrated <= 0:
        raise ValueError("alpha_uas and alpha_calibrated must both be > 0.")


def sample_standard_error(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def compute_system_summary(cfg: ExperimentConfig) -> dict[str, object]:
    mu = np.asarray(cfg.system.service_rates, dtype=np.float64)
    total_capacity = float(np.sum(mu))
    rho = float(cfg.system.arrival_rate / total_capacity)
    return {
        "num_servers": int(cfg.system.num_servers),
        "arrival_rate": float(cfg.system.arrival_rate),
        "service_rates": mu.tolist(),
        "total_capacity": total_capacity,
        "rho": rho,
    }


def compute_scuas_constants(
    *,
    service_rates: Sequence[float],
    arrival_rate: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> dict[str, object]:
    _validate_candidate_values(beta=beta, gamma=gamma, c=c)
    mu = np.asarray(service_rates, dtype=np.float64)
    if mu.ndim != 1 or mu.size == 0:
        raise ValueError("service_rates must be a non-empty one-dimensional array.")
    if np.any(mu <= 0):
        raise ValueError("All service rates must be strictly positive.")
    if arrival_rate <= 0:
        raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    prior_weights = np.power(mu, gamma) * np.exp(-alpha * c / np.power(mu, beta))
    z_value = float(np.sum(prior_weights))
    if not math.isfinite(z_value) or z_value <= 0:
        raise ValueError(f"Invalid normalization constant Z={z_value}")

    a_vector = (
        np.power(mu, gamma - beta)
        * np.exp(-alpha * c / np.power(mu, beta))
        / z_value
    )
    service_term = np.power(mu, 1.0 - beta)
    drift_coeffs = service_term - arrival_rate * a_vector
    epsilon = float(np.min(drift_coeffs))
    r_value = float(
        0.5 * arrival_rate * np.max(np.power(mu, -beta))
        + 0.5 * np.sum(service_term)
    )

    return {
        "beta": float(beta),
        "gamma": float(gamma),
        "c": float(c),
        "alpha": float(alpha),
        "Z": z_value,
        "a_vector": a_vector.tolist(),
        "service_term": service_term.tolist(),
        "drift_coefficients": drift_coeffs.tolist(),
        "epsilon": epsilon,
        "R": r_value,
        "is_certified": bool(epsilon > 0.0),
    }


def audit_candidates(
    *,
    cfg: ExperimentConfig,
    candidates: Sequence[CandidateSpec],
    alpha_calibrated: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        constants = compute_scuas_constants(
            service_rates=cfg.system.service_rates,
            arrival_rate=cfg.system.arrival_rate,
            alpha=alpha_calibrated,
            beta=candidate.beta,
            gamma=candidate.gamma,
            c=candidate.c,
        )
        rows.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "notes": candidate.notes,
                **constants,
            }
        )
    return rows


def certified_scuas_candidates(
    audit_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        row
        for row in audit_rows
        if row["family"] == "scuas_candidate" and bool(row["is_certified"])
    ]


def evaluate_policy(
    *,
    label: str,
    family: str,
    policy,
    cfg: ExperimentConfig,
) -> dict[str, object]:
    burn_in = cfg.simulation.burn_in_fraction
    results = run_replications(
        num_servers=cfg.system.num_servers,
        arrival_rate=cfg.system.arrival_rate,
        service_rates=np.asarray(cfg.system.service_rates, dtype=np.float64),
        policy=policy,
        num_replications=cfg.simulation.num_replications,
        sim_time=cfg.simulation.ssa.sim_time,
        sample_interval=cfg.simulation.ssa.sample_interval,
        base_seed=cfg.simulation.seed,
        progress_desc=f"scuas policy eval ({label})",
    )

    q_totals = [float(time_averaged_queue_lengths(r, burn_in).sum()) for r in results]
    ginis = [float(gini_coefficient(time_averaged_queue_lengths(r, burn_in))) for r in results]
    sojourns = [
        float(sojourn_time_estimate(r, cfg.system.arrival_rate, burn_in))
        for r in results
    ]

    return {
        "label": label,
        "family": family,
        "mean_q_total": float(np.mean(q_totals)),
        "se_q_total": sample_standard_error(q_totals),
        "mean_gini": float(np.mean(ginis)),
        "se_gini": sample_standard_error(ginis),
        "mean_sojourn": float(np.mean(sojourns)),
        "se_sojourn": sample_standard_error(sojourns),
        "num_replications": int(cfg.simulation.num_replications),
        "sim_time": float(cfg.simulation.ssa.sim_time),
        "sample_interval": float(cfg.simulation.ssa.sample_interval),
        "burn_in_fraction": float(cfg.simulation.burn_in_fraction),
        "base_seed": int(cfg.simulation.seed),
    }


def build_policy_suite(
    *,
    cfg: ExperimentConfig,
    audit_rows: Sequence[dict[str, object]],
    protocol: ValidationProtocol,
) -> list[tuple[str, str, object, dict[str, object] | None]]:
    mu = np.asarray(cfg.system.service_rates, dtype=np.float64)
    suite: list[tuple[str, str, object, dict[str, object] | None]] = [
        (
            "UAS",
            "uas_baseline",
            UASRouting(mu=mu, alpha=protocol.alpha_uas),
            None,
        ),
        (
            "Calibrated UAS (empirical default)",
            "empirical_calibrated_default",
            CalibratedUASRouting(
                mu=mu,
                alpha=protocol.alpha_calibrated,
                beta=0.85,
                gamma=0.5,
                c=0.5,
            ),
            next(row for row in audit_rows if row["name"] == "calibrated_default"),
        ),
    ]

    for row in certified_scuas_candidates(audit_rows):
        label = (
            f"SCUAS (beta={row['beta']}, gamma={row['gamma']}, c={row['c']})"
        )
        suite.append(
            (
                label,
                "scuas_certified",
                CalibratedUASRouting(
                    mu=mu,
                    alpha=protocol.alpha_calibrated,
                    beta=float(row["beta"]),
                    gamma=float(row["gamma"]),
                    c=float(row["c"]),
                ),
                row,
            )
        )

    return suite


def render_summary_markdown(
    *,
    system_summary: dict[str, object],
    protocol: ValidationProtocol,
    audit_rows: Sequence[dict[str, object]],
    rerun_rows: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# SCUAS Validation Summary",
        "",
        "## Benchmark Setup",
        f"- num_servers: {system_summary['num_servers']}",
        f"- arrival_rate: {system_summary['arrival_rate']}",
        f"- total_capacity: {system_summary['total_capacity']}",
        f"- rho: {system_summary['rho']}",
        f"- service_rates: {system_summary['service_rates']}",
        "",
        "## Protocol",
        f"- num_replications: {protocol.num_replications}",
        f"- sim_time: {protocol.sim_time}",
        f"- sample_interval: {protocol.sample_interval}",
        f"- burn_in_fraction: {protocol.burn_in_fraction}",
        f"- base_seed: {protocol.base_seed}",
        f"- alpha_uas: {protocol.alpha_uas}",
        f"- alpha_calibrated: {protocol.alpha_calibrated}",
        "",
        "## Audit",
    ]
    for row in audit_rows:
        lines.append(
            "- "
            f"{row['name']}: epsilon={row['epsilon']:.12f}, "
            f"R={row['R']:.12f}, certified={row['is_certified']}"
        )

    if rerun_rows:
        lines.extend(["", "## Rerun"])
        for row in rerun_rows:
            lines.append(
                "- "
                f"{row['label']}: "
                f"E[Q_total]={row['mean_q_total']:.6f} +/- {row['se_q_total']:.6f}, "
                f"Gini={row['mean_gini']:.6f} +/- {row['se_gini']:.6f}, "
                f"Sojourn={row['mean_sojourn']:.6f} +/- {row['se_sojourn']:.6f}"
            )

    return "\n".join(lines) + "\n"


def run_audit(
    *,
    cfg: ExperimentConfig,
    run_dir: Path,
    candidates: Sequence[CandidateSpec],
    protocol: ValidationProtocol,
) -> list[dict[str, object]]:
    audit_rows = audit_candidates(
        cfg=cfg,
        candidates=candidates,
        alpha_calibrated=protocol.alpha_calibrated,
    )
    audit_log_path = metrics_path(run_dir, "scuas_audit.jsonl")
    for row in audit_rows:
        append_metrics_jsonl(row, audit_log_path)

    summary_payload = {
        "system": compute_system_summary(cfg),
        "protocol": asdict(protocol),
        "audit_rows": audit_rows,
    }
    metadata_path(run_dir, "scuas_audit_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    return audit_rows


def run_rerun(
    *,
    cfg: ExperimentConfig,
    run_dir: Path,
    audit_rows: Sequence[dict[str, object]],
    protocol: ValidationProtocol,
) -> list[dict[str, object]]:
    suite = build_policy_suite(cfg=cfg, audit_rows=audit_rows, protocol=protocol)
    rerun_rows: list[dict[str, object]] = []
    rerun_log_path = metrics_path(run_dir, "scuas_policy_comparison.jsonl")

    if not certified_scuas_candidates(audit_rows):
        log.warning(
            "No SCUAS candidates satisfied epsilon > 0 under the active benchmark. "
            "Rerun will include only UAS and the empirical calibrated default."
        )

    for label, family, policy, audit_row in suite:
        log.info("Evaluating %s", label)
        metrics = evaluate_policy(label=label, family=family, policy=policy, cfg=cfg)
        if audit_row is not None:
            metrics.update(
                {
                    "beta": audit_row["beta"],
                    "gamma": audit_row["gamma"],
                    "c": audit_row["c"],
                    "epsilon": audit_row["epsilon"],
                    "certified": audit_row["is_certified"],
                }
            )
        append_metrics_jsonl(metrics, rerun_log_path)
        rerun_rows.append(metrics)

    metadata_path(run_dir, "scuas_policy_summary.json").write_text(
        json.dumps(
            {
                "system": compute_system_summary(cfg),
                "protocol": asdict(protocol),
                "rerun_rows": rerun_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rerun_rows


def write_summary_files(
    *,
    cfg: ExperimentConfig,
    run_dir: Path,
    protocol: ValidationProtocol,
    audit_rows: Sequence[dict[str, object]],
    rerun_rows: Sequence[dict[str, object]],
) -> None:
    summary_md = render_summary_markdown(
        system_summary=compute_system_summary(cfg),
        protocol=protocol,
        audit_rows=audit_rows,
        rerun_rows=rerun_rows,
    )
    metadata_path(run_dir, "scuas_validation_summary.md").write_text(
        summary_md,
        encoding="utf-8",
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the mandatory SCUAS theorem audit and the calibrated-only rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["audit", "rerun", "full"],
        default=DEFAULT_MODE,
        help="Which validation stage to execute.",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Config profile to resolve before applying anchor protocol knobs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/post_scuas_validation",
        help="Root output directory for validation capsules.",
    )
    parser.add_argument(
        "--num-replications",
        type=int,
        default=ANCHOR_REPLICATIONS,
        help="Replication count for rerun comparisons.",
    )
    parser.add_argument(
        "--sim-time",
        type=float,
        default=ANCHOR_SIM_TIME,
        help="Simulation horizon for rerun comparisons.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=ANCHOR_SAMPLE_INTERVAL,
        help="Sampling interval for SSA reruns.",
    )
    parser.add_argument(
        "--burn-in-fraction",
        type=float,
        default=ANCHOR_BURN_IN_FRACTION,
        help="Burn-in fraction used for steady-state metrics.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=ANCHOR_BASE_SEED,
        help="Base seed for all replication runs.",
    )
    parser.add_argument(
        "--alpha-uas",
        type=float,
        default=DEFAULT_UAS_ALPHA,
        help="UAS alpha used by the anchor publication comparison.",
    )
    parser.add_argument(
        "--alpha-calibrated",
        type=float,
        default=DEFAULT_CALIBRATED_ALPHA,
        help="Alpha used for empirical Calibrated UAS and SCUAS candidates.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Additional SCUAS candidate triplet formatted as beta,gamma,c.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional OmegaConf dotlist overrides, e.g. system.service_rates=[...]",
    )
    return parser


def build_protocol(args: argparse.Namespace) -> ValidationProtocol:
    return ValidationProtocol(
        num_replications=args.num_replications,
        sim_time=args.sim_time,
        sample_interval=args.sample_interval,
        burn_in_fraction=args.burn_in_fraction,
        base_seed=args.base_seed,
        alpha_uas=args.alpha_uas,
        alpha_calibrated=args.alpha_calibrated,
    )


def log_audit_table(rows: Sequence[dict[str, object]]) -> None:
    log.info("SCUAS theorem audit:")
    for row in rows:
        log.info(
            "  %s | family=%s | beta=%.4f gamma=%.4f c=%.4f | epsilon=%.12f | certified=%s",
            row["name"],
            row["family"],
            row["beta"],
            row["gamma"],
            row["c"],
            row["epsilon"],
            row["is_certified"],
        )


def log_rerun_table(rows: Sequence[dict[str, object]]) -> None:
    log.info("Policy rerun summary:")
    for row in rows:
        log.info(
            "  %s | E[Q_total]=%.6f +/- %.6f | Gini=%.6f +/- %.6f | Sojourn=%.6f +/- %.6f",
            row["label"],
            row["mean_q_total"],
            row["se_q_total"],
            row["mean_gini"],
            row["se_gini"],
            row["mean_sojourn"],
            row["se_sojourn"],
        )


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    protocol = build_protocol(args)

    cfg, resolved_raw = load_policy_experiment_config(
        config_name=args.config_name,
        overrides=args.overrides,
        protocol=protocol,
        output_dir=args.output_dir,
    )
    candidates = extend_candidate_catalog(args.candidate)

    run_dir, run_id = get_run_config(cfg, "scuas_validation", resolved_raw)
    log.info("SCUAS validation capsule: %s", run_dir)
    log.info("Run id: %s", run_id)

    audit_rows: list[dict[str, object]] = []
    rerun_rows: list[dict[str, object]] = []

    if args.mode in {"audit", "full", "rerun"}:
        audit_rows = run_audit(
            cfg=cfg,
            run_dir=run_dir,
            candidates=candidates,
            protocol=protocol,
        )
        log_audit_table(audit_rows)
        if not certified_scuas_candidates(audit_rows):
            log.warning(
                "Audit found no theorem-certified SCUAS candidates for the active setup."
            )

    if args.mode in {"rerun", "full"}:
        rerun_rows = run_rerun(
            cfg=cfg,
            run_dir=run_dir,
            audit_rows=audit_rows,
            protocol=protocol,
        )
        log_rerun_table(rerun_rows)

    write_summary_files(
        cfg=cfg,
        run_dir=run_dir,
        protocol=protocol,
        audit_rows=audit_rows,
        rerun_rows=rerun_rows,
    )
    log.info("Validation outputs written under %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
