#!/usr/bin/env python3
"""
Validation capsule for the direct CTMC quadratic proof route.

This script validates the newer theorem attempt based on the weighted
quadratic Lyapunov function and the exact softmax-minimum bound.

What it computes:
    - theorem constants for audited candidates
    - sampled drift-bound checks on a reproducible state bank
    - benchmark rerun metrics for audited candidates

What it does not claim:
    - that the full stochastic theorem is promoted beyond the current `z2`
      status files
    - that sampled support alone is a final theorem

Modes
-----
`audit`
    Compute the theorem constants for the active benchmark candidates and check
    the exact generator inequality on a sampled state bank.
`rerun`
    Rerun benchmark policy comparisons for the audited candidates.
`full`
    Run both audit and rerun.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gibbsq.experiments.verification import reflected_uas_proof_search as ps  # noqa: E402
from studies.analysis.common.metrics import (  # noqa: E402
    gini_coefficient,
    sojourn_time_estimate,
    time_averaged_queue_lengths,
)
from gibbsq.qroute.core.config import (  # noqa: E402
    ExperimentConfig,
    _profile_path,
    hydra_to_config,
    load_experiment_config,
    validate,
)
from gibbsq.qroute.core.policies import ReflectedUASRouting, UASRouting  # noqa: E402
from gibbsq.qroute.engines.numpy_engine import run_replications  # noqa: E402
from gibbsq.qroute.utils.exporter import append_metrics_jsonl  # noqa: E402
from gibbsq.qroute.utils.logging import get_run_config  # noqa: E402
from gibbsq.qroute.utils.run_artifacts import metadata_path, metrics_path  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "final_experiment"
DEFAULT_MODE = "full"
DEFAULT_OUTPUT_DIR = "outputs/direct_ctmc_validation"

ANCHOR_REPLICATIONS = 32
ANCHOR_SIM_TIME = 15000.0
ANCHOR_SAMPLE_INTERVAL = 1.0
ANCHOR_BURN_IN_FRACTION = 0.2
ANCHOR_BASE_SEED = 42
DEFAULT_UAS_ALPHA = 10.0
DEFAULT_REFLECTED_ALPHA = 20.0

DEFAULT_AUDIT_TRAJECTORY_REPS = 4
DEFAULT_AUDIT_TRAJECTORY_SIM_TIME = 1000.0
DEFAULT_AUDIT_TRAJECTORY_SAMPLE_INTERVAL = 5.0


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    beta: float
    gamma: float
    c: float
    family: str
    alpha_kind: str = "reflected"
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
    alpha_reflected: float = DEFAULT_REFLECTED_ALPHA
    audit_trajectory_reps: int = DEFAULT_AUDIT_TRAJECTORY_REPS
    audit_trajectory_sim_time: float = DEFAULT_AUDIT_TRAJECTORY_SIM_TIME
    audit_trajectory_sample_interval: float = DEFAULT_AUDIT_TRAJECTORY_SAMPLE_INTERVAL


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def _validate_candidate_values(*, beta: float, gamma: float, c: float) -> None:
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if c < 0:
        raise ValueError(f"c must be >= 0, got {c}")
    if not math.isfinite(beta) or not math.isfinite(gamma) or not math.isfinite(c):
        raise ValueError("Candidate parameters must be finite real values.")


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


def default_candidate_catalog() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="uas_special_case",
            beta=1.0,
            gamma=1.0,
            c=1.0,
            family="uas_special_case",
            alpha_kind="uas",
            notes="UAS written as a Reflected-UAS special case for theorem audit.",
        ),
        CandidateSpec(
            name="reflected_default",
            beta=0.85,
            gamma=0.5,
            c=0.5,
            family="benchmark_default",
            alpha_kind="reflected",
            notes="Current manuscript default empirical reflected policy.",
        ),
        CandidateSpec(
            name="grid_b0p5_g0p25_c0p25",
            beta=0.5,
            gamma=0.25,
            c=0.25,
            family="grid_candidate",
            alpha_kind="reflected",
            notes="Auxiliary grid candidate.",
        ),
        CandidateSpec(
            name="grid_b0p5_g0p5_c0p25",
            beta=0.5,
            gamma=0.5,
            c=0.25,
            family="grid_candidate",
            alpha_kind="reflected",
            notes="Auxiliary grid candidate.",
        ),
        CandidateSpec(
            name="grid_b0p7_g0p25_c0p25",
            beta=0.7,
            gamma=0.25,
            c=0.25,
            family="grid_candidate",
            alpha_kind="reflected",
            notes="Auxiliary grid candidate.",
        ),
    ]


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
                family="user_candidate",
                alpha_kind="reflected",
                notes="User-specified reflected-UAS candidate.",
            )
        )
        known.add(triplet)
    return catalog


def build_protocol(args: argparse.Namespace) -> ValidationProtocol:
    return ValidationProtocol(
        num_replications=args.num_replications,
        sim_time=args.sim_time,
        sample_interval=args.sample_interval,
        burn_in_fraction=args.burn_in_fraction,
        base_seed=args.base_seed,
        alpha_uas=args.alpha_uas,
        alpha_reflected=args.alpha_reflected,
        audit_trajectory_reps=args.audit_trajectory_reps,
        audit_trajectory_sim_time=args.audit_trajectory_sim_time,
        audit_trajectory_sample_interval=args.audit_trajectory_sample_interval,
    )


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
    return cfg, resolved_raw


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


def candidate_alpha(candidate: CandidateSpec, protocol: ValidationProtocol) -> float:
    if candidate.alpha_kind == "uas":
        return float(protocol.alpha_uas)
    return float(protocol.alpha_reflected)


def compute_direct_ctmc_constants(
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
    if np.any(mu <= 0.0) or not np.all(np.isfinite(mu)):
        raise ValueError("All service rates must be finite and strictly positive.")
    if arrival_rate <= 0.0 or not math.isfinite(arrival_rate):
        raise ValueError(f"arrival_rate must be finite and > 0, got {arrival_rate}")
    if alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite and > 0, got {alpha}")

    lambda_value = float(arrival_rate)
    lambda_total = float(np.sum(mu))
    mu_beta = np.power(mu, beta)
    service_term = np.power(mu, 1.0 - beta)
    kappa = (c / mu_beta) - (gamma / alpha) * np.log(mu)
    c1 = float(math.log(mu.size) / alpha + np.max(kappa) - np.min(kappa))
    c0 = float(0.5 * lambda_value * np.max(np.power(mu, -beta)) + 0.5 * np.sum(service_term))
    r_value = float(lambda_value * c1 + c0)
    epsilon = float(
        min(
            (lambda_total - lambda_value) / float(np.sum(mu_beta)),
            float(np.min(service_term)),
        )
    )
    load_margin = float(lambda_total - lambda_value)
    theorem_load_ok = bool(load_margin > 0.0)
    has_positive_epsilon = bool(epsilon > 0.0)
    return {
        "beta": float(beta),
        "gamma": float(gamma),
        "c": float(c),
        "alpha": float(alpha),
        "kappa": kappa.tolist(),
        "service_term": service_term.tolist(),
        "sum_mu_beta": float(np.sum(mu_beta)),
        "load_margin": load_margin,
        "epsilon": epsilon,
        "R": r_value,
        "load_condition_ok": theorem_load_ok,
        "has_positive_epsilon": has_positive_epsilon,
    }


def exact_quadratic_generator_drift(
    states: np.ndarray,
    *,
    mu: np.ndarray,
    arrival_rate: float,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> np.ndarray:
    states_i = np.asarray(states, dtype=np.int64)
    if states_i.ndim != 2:
        raise ValueError("states must have shape (M, N)")
    weights = np.power(mu[None, :], -beta)
    probs = ps.reflected_policy_probs(
        states_i,
        mu=mu,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        c=c,
    )
    q = states_i.astype(np.float64)
    arrival_term = arrival_rate * np.sum(probs * weights * (q + 0.5), axis=1)
    departure_term = np.sum(mu[None, :] * (q > 0.0) * (-weights * (q - 0.5)), axis=1)
    return arrival_term + departure_term


def audit_candidate_row(
    *,
    cfg: ExperimentConfig,
    candidate: CandidateSpec,
    protocol: ValidationProtocol,
) -> dict[str, object]:
    alpha = candidate_alpha(candidate, protocol)
    constants = compute_direct_ctmc_constants(
        service_rates=cfg.system.service_rates,
        arrival_rate=cfg.system.arrival_rate,
        alpha=alpha,
        beta=candidate.beta,
        gamma=candidate.gamma,
        c=candidate.c,
    )
    policy_point = ps.PolicyPoint(
        name=candidate.name,
        beta=float(candidate.beta),
        gamma=float(candidate.gamma),
        c=float(candidate.c),
        alpha=float(alpha),
        family=str(candidate.family),
    )
    states = ps.build_state_bank(
        cfg=cfg,
        seed=protocol.base_seed,
        include_trajectories=True,
        trajectory_reps=protocol.audit_trajectory_reps,
        trajectory_sim_time=protocol.audit_trajectory_sim_time,
        trajectory_sample_interval=protocol.audit_trajectory_sample_interval,
        trajectory_policy=policy_point,
    )
    mu = np.asarray(cfg.system.service_rates, dtype=np.float64)
    drifts = exact_quadratic_generator_drift(
        states,
        mu=mu,
        arrival_rate=cfg.system.arrival_rate,
        alpha=alpha,
        beta=candidate.beta,
        gamma=candidate.gamma,
        c=candidate.c,
    )
    norms = states.sum(axis=1).astype(np.float64)
    theorem_rhs = -float(constants["epsilon"]) * norms + float(constants["R"])
    residual = drifts - theorem_rhs
    max_violation_idx = int(np.argmax(residual))
    max_violation = float(residual[max_violation_idx])
    positive_violation_count = int(np.count_nonzero(residual > 1e-9))

    return {
        "name": candidate.name,
        "family": candidate.family,
        "notes": candidate.notes,
        **constants,
        "state_bank_size": int(states.shape[0]),
        "max_sampled_drift": float(np.max(drifts)),
        "min_sampled_drift": float(np.min(drifts)),
        "max_sampled_residual": max_violation,
        "positive_violation_count": positive_violation_count,
        "passes_sampled_bound": bool(positive_violation_count == 0),
        "worst_state": states[max_violation_idx].tolist(),
        "worst_state_drift": float(drifts[max_violation_idx]),
        "worst_state_rhs": float(theorem_rhs[max_violation_idx]),
    }


def audit_candidates(
    *,
    cfg: ExperimentConfig,
    candidates: Sequence[CandidateSpec],
    protocol: ValidationProtocol,
) -> list[dict[str, object]]:
    return [
        audit_candidate_row(cfg=cfg, candidate=candidate, protocol=protocol)
        for candidate in candidates
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
        progress_desc=f"direct ctmc policy eval ({label})",
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
        ("UAS", "uas_baseline", UASRouting(mu=mu, alpha=protocol.alpha_uas), None),
        (
            "Reflected UAS (empirical default)",
            "benchmark_default",
            ReflectedUASRouting(
                mu=mu,
                alpha=protocol.alpha_reflected,
                beta=0.85,
                gamma=0.5,
                c=0.5,
            ),
            next(row for row in audit_rows if row["name"] == "reflected_default"),
        ),
    ]
    for row in audit_rows:
        if row["name"] in {"uas_special_case", "reflected_default"}:
            continue
        suite.append(
            (
                f"Candidate (beta={row['beta']}, gamma={row['gamma']}, c={row['c']})",
                str(row["family"]),
                ReflectedUASRouting(
                    mu=mu,
                    alpha=float(row["alpha"]),
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
        "# Direct CTMC Validation Summary",
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
        f"- alpha_reflected: {protocol.alpha_reflected}",
        f"- audit_trajectory_reps: {protocol.audit_trajectory_reps}",
        f"- audit_trajectory_sim_time: {protocol.audit_trajectory_sim_time}",
        f"- audit_trajectory_sample_interval: {protocol.audit_trajectory_sample_interval}",
        "",
        "## Audit",
    ]
    for row in audit_rows:
        lines.append(
            "- "
            f"{row['name']}: epsilon={row['epsilon']:.12f}, "
            f"R={row['R']:.12f}, "
            f"load_ok={row['load_condition_ok']}, "
            f"positive_epsilon={row['has_positive_epsilon']}, "
            f"sampled_bound_pass={row['passes_sampled_bound']}, "
            f"max_residual={row['max_sampled_residual']:.12e}"
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
    audit_rows = audit_candidates(cfg=cfg, candidates=candidates, protocol=protocol)
    audit_log_path = metrics_path(run_dir, "direct_ctmc_audit.jsonl")
    for row in audit_rows:
        append_metrics_jsonl(row, audit_log_path)
    metadata_path(run_dir, "direct_ctmc_audit_summary.json").write_text(
        json.dumps(
            {
                "system": compute_system_summary(cfg),
                "protocol": asdict(protocol),
                "audit_rows": audit_rows,
            },
            indent=2,
        ),
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
    rerun_log_path = metrics_path(run_dir, "direct_ctmc_policy_comparison.jsonl")

    for label, family, policy, audit_row in suite:
        log.info("Evaluating %s", label)
        metrics = evaluate_policy(label=label, family=family, policy=policy, cfg=cfg)
        if audit_row is not None:
            metrics.update(
                {
                    "beta": audit_row["beta"],
                    "gamma": audit_row["gamma"],
                    "c": audit_row["c"],
                    "alpha": audit_row["alpha"],
                    "epsilon": audit_row["epsilon"],
                    "R": audit_row["R"],
                    "load_condition_ok": audit_row["load_condition_ok"],
                    "has_positive_epsilon": audit_row["has_positive_epsilon"],
                    "passes_sampled_bound": audit_row["passes_sampled_bound"],
                }
            )
        append_metrics_jsonl(metrics, rerun_log_path)
        rerun_rows.append(metrics)

    metadata_path(run_dir, "direct_ctmc_policy_summary.json").write_text(
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


def log_audit_table(rows: Sequence[dict[str, object]]) -> None:
    log.info("Direct CTMC theorem audit:")
    for row in rows:
        log.info(
            "  %s | family=%s | beta=%.4f gamma=%.4f c=%.4f alpha=%.4f | epsilon=%.12f | sampled_pass=%s | max_residual=%.12e",
            row["name"],
            row["family"],
            row["beta"],
            row["gamma"],
            row["c"],
            row["alpha"],
            row["epsilon"],
            row["passes_sampled_bound"],
            row["max_sampled_residual"],
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the direct CTMC quadratic proof route and rerun benchmark comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["audit", "rerun", "full"], default=DEFAULT_MODE)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-replications", type=int, default=ANCHOR_REPLICATIONS)
    parser.add_argument("--sim-time", type=float, default=ANCHOR_SIM_TIME)
    parser.add_argument("--sample-interval", type=float, default=ANCHOR_SAMPLE_INTERVAL)
    parser.add_argument("--burn-in-fraction", type=float, default=ANCHOR_BURN_IN_FRACTION)
    parser.add_argument("--base-seed", type=int, default=ANCHOR_BASE_SEED)
    parser.add_argument("--alpha-uas", type=float, default=DEFAULT_UAS_ALPHA)
    parser.add_argument("--alpha-reflected", type=float, default=DEFAULT_REFLECTED_ALPHA)
    parser.add_argument("--audit-trajectory-reps", type=int, default=DEFAULT_AUDIT_TRAJECTORY_REPS)
    parser.add_argument("--audit-trajectory-sim-time", type=float, default=DEFAULT_AUDIT_TRAJECTORY_SIM_TIME)
    parser.add_argument("--audit-trajectory-sample-interval", type=float, default=DEFAULT_AUDIT_TRAJECTORY_SAMPLE_INTERVAL)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Additional reflected candidate triplet formatted as beta,gamma,c.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser


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
    run_dir, run_id = get_run_config(cfg, "direct_ctmc_validation", resolved_raw)
    log.info("Direct CTMC validation capsule: %s", run_dir)
    log.info("Run id: %s", run_id)

    audit_rows: list[dict[str, object]] = []
    rerun_rows: list[dict[str, object]] = []

    if args.mode in {"audit", "full"}:
        audit_rows = run_audit(cfg=cfg, run_dir=run_dir, candidates=candidates, protocol=protocol)
        log_audit_table(audit_rows)

    if args.mode in {"rerun", "full"}:
        if not audit_rows:
            audit_rows = run_audit(cfg=cfg, run_dir=run_dir, candidates=candidates, protocol=protocol)
        rerun_rows = run_rerun(cfg=cfg, run_dir=run_dir, audit_rows=audit_rows, protocol=protocol)
        log_rerun_table(rerun_rows)

    metadata_path(run_dir, "direct_ctmc_validation_summary.md").write_text(
        render_summary_markdown(
            system_summary=compute_system_summary(cfg),
            protocol=protocol,
            audit_rows=audit_rows,
            rerun_rows=rerun_rows,
        ),
        encoding="utf-8",
    )
    log.info("Validation outputs written under %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


