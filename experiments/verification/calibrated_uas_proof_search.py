#!/usr/bin/env python3
"""
Exploratory proof-search workflow for the full Calibrated UAS family.

This script is intentionally honest about what it can and cannot do.
It evaluates candidate Lyapunov/proof templates numerically against the
exact generator under the calibrated routing law and ranks them by how
well they achieve negative drift on sampled state sets. A good score is
evidence of a promising proof direction, not a formal theorem.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linprog

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gibbsq.core.config import ExperimentConfig, _profile_path, hydra_to_config, load_experiment_config, validate  # noqa: E402
from gibbsq.core.policies import CalibratedUASRouting  # noqa: E402
from gibbsq.engines.numpy_engine import run_replications  # noqa: E402
from gibbsq.utils.logging import get_run_config  # noqa: E402
from gibbsq.utils.run_artifacts import metadata_path, metrics_path  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "final_experiment"
DEFAULT_OUTPUT_DIR = "outputs/calibrated_uas_proof_search"
DEFAULT_TAIL_NORM = 20.0
DEFAULT_TRAJ_REPS = 4
DEFAULT_TRAJ_SIM_TIME = 1000.0
DEFAULT_TRAJ_SAMPLE_INTERVAL = 5.0


@dataclass(frozen=True)
class PolicyPoint:
    name: str
    beta: float
    gamma: float
    c: float
    alpha: float
    family: str


@dataclass(frozen=True)
class CandidateTemplate:
    name: str
    strategy: str
    family: str
    eta: float
    p: float | None = None
    theta: float | None = None
    mix: float | None = None


@dataclass(frozen=True)
class TemplateEvaluation:
    row: dict[str, object]
    drifts: np.ndarray


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def load_policy_cfg(
    *,
    config_name: str,
    overrides: Sequence[str],
    output_dir: str,
) -> tuple[ExperimentConfig, DictConfig]:
    raw_profile = OmegaConf.load(_profile_path(config_name))
    if overrides:
        raw_profile = OmegaConf.merge(raw_profile, OmegaConf.from_dotlist(list(overrides)))
    cfg, resolved_raw = load_experiment_config(raw_profile, "policy", profile_name=config_name)
    OmegaConf.update(resolved_raw, "wandb.enabled", False)
    OmegaConf.update(resolved_raw, "output_dir", output_dir)
    cfg = hydra_to_config(resolved_raw)
    validate(cfg)
    return cfg, resolved_raw


def calibrated_policy_probs(
    states: np.ndarray,
    *,
    mu: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    c: float,
) -> np.ndarray:
    q = states.astype(np.float64)
    logits = gamma * np.log(mu)[None, :]
    logits = logits - alpha * ((q + c) / (mu[None, :] ** beta))
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def evaluate_template_batch(
    states: np.ndarray,
    *,
    mu: np.ndarray,
    template: CandidateTemplate,
) -> np.ndarray:
    q = states.astype(np.float64)
    weights = np.power(mu[None, :], -template.eta)

    if template.family == "linear_sum":
        return np.sum(weights * q, axis=1)

    if template.family == "quadratic_sum":
        return 0.5 * np.sum(weights * q * q, axis=1)

    if template.family == "power_sum":
        assert template.p is not None
        return np.sum(weights * np.power(q, template.p), axis=1)

    scaled = weights * q

    if template.family == "exponential_sum":
        assert template.theta is not None
        return np.sum(np.exp(template.theta * scaled) - 1.0, axis=1)

    if template.family == "logsumexp":
        assert template.theta is not None
        logits = template.theta * scaled
        m = logits.max(axis=1, keepdims=True)
        stabilized = np.exp(logits - m)
        return (
            (np.log(stabilized.sum(axis=1)) + m[:, 0]) / template.theta
        )

    if template.family == "max_power":
        exponent = 1.0 if template.p is None else template.p
        return np.max(np.power(scaled, exponent), axis=1)

    if template.family == "variance_quadratic":
        centered = scaled - scaled.mean(axis=1, keepdims=True)
        return np.sum(centered * centered, axis=1)

    if template.family == "hybrid_quadratic_variance":
        centered = scaled - scaled.mean(axis=1, keepdims=True)
        base = 0.5 * np.sum(weights * q * q, axis=1)
        mix = 1.0 if template.mix is None else template.mix
        return base + mix * np.sum(centered * centered, axis=1)

    if template.family == "total_load_power":
        exponent = 2.0 if template.p is None else template.p
        total = np.sum(scaled, axis=1)
        return np.power(total, exponent)

    raise ValueError(f"Unknown template family: {template.family}")


def exact_generator_drift(
    states: np.ndarray,
    *,
    mu: np.ndarray,
    arrival_rate: float,
    policy_point: PolicyPoint,
    template: CandidateTemplate,
) -> np.ndarray:
    states_i = np.asarray(states, dtype=np.int64)
    if states_i.ndim != 2:
        raise ValueError("states must have shape (M, N)")
    num_states, num_servers = states_i.shape

    v_current = evaluate_template_batch(states_i, mu=mu, template=template)

    plus_states = np.repeat(states_i[:, None, :], num_servers, axis=1)
    plus_states[np.arange(num_states)[:, None], np.arange(num_servers)[None, :], np.arange(num_servers)[None, :]] += 1
    plus_states_flat = plus_states.reshape(num_states * num_servers, num_servers)
    v_plus = evaluate_template_batch(plus_states_flat, mu=mu, template=template).reshape(num_states, num_servers)

    minus_states = np.repeat(states_i[:, None, :], num_servers, axis=1)
    mask = states_i > 0
    valid_rows, valid_cols = np.nonzero(mask)
    minus_states[valid_rows, valid_cols, valid_cols] -= 1
    minus_states_flat = minus_states.reshape(num_states * num_servers, num_servers)
    v_minus = evaluate_template_batch(minus_states_flat, mu=mu, template=template).reshape(num_states, num_servers)

    probs = calibrated_policy_probs(
        states_i,
        mu=mu,
        alpha=policy_point.alpha,
        beta=policy_point.beta,
        gamma=policy_point.gamma,
        c=policy_point.c,
    )

    arrival_term = arrival_rate * np.sum(probs * (v_plus - v_current[:, None]), axis=1)
    departure_term = np.sum(
        mu[None, :] * mask.astype(np.float64) * (v_minus - v_current[:, None]),
        axis=1,
    )
    return arrival_term + departure_term


def enumerate_templates() -> list[CandidateTemplate]:
    templates: list[CandidateTemplate] = []

    for eta in (0.0, 0.5, 0.85, 1.0):
        templates.append(
            CandidateTemplate(
                name=f"quadratic_eta_{eta:g}",
                strategy="weighted_quadratic",
                family="quadratic_sum",
                eta=eta,
            )
        )

    for p in (1.5, 2.5, 3.0):
        for eta in (0.0, 0.5, 1.0):
            templates.append(
                CandidateTemplate(
                    name=f"power_p_{p:g}_eta_{eta:g}",
                    strategy="power_sum",
                    family="power_sum",
                    eta=eta,
                    p=p,
                )
            )

    for eta in (0.0, 0.5, 1.0):
        templates.append(
            CandidateTemplate(
                name=f"linear_eta_{eta:g}",
                strategy="fluid_weighted_l1",
                family="linear_sum",
                eta=eta,
            )
        )

    for theta in (0.02, 0.05, 0.1):
        for eta in (0.0, 0.5, 1.0):
            templates.append(
                CandidateTemplate(
                    name=f"exp_theta_{theta:g}_eta_{eta:g}",
                    strategy="tilted_exponential",
                    family="exponential_sum",
                    eta=eta,
                    theta=theta,
                )
            )
            templates.append(
                CandidateTemplate(
                    name=f"logsumexp_theta_{theta:g}_eta_{eta:g}",
                    strategy="smoothed_max",
                    family="logsumexp",
                    eta=eta,
                    theta=theta,
                )
            )

    for p in (1.0, 2.0):
        for eta in (0.0, 0.5, 1.0):
            templates.append(
                CandidateTemplate(
                    name=f"max_p_{p:g}_eta_{eta:g}",
                    strategy="state_partition_max",
                    family="max_power",
                    eta=eta,
                    p=p,
                )
            )

    for eta in (0.0, 0.5, 1.0):
        templates.append(
            CandidateTemplate(
                name=f"variance_eta_{eta:g}",
                strategy="imbalance_variance",
                family="variance_quadratic",
                eta=eta,
            )
        )

    for mix in (0.5, 1.0, 2.0):
        for eta in (0.0, 0.5, 1.0):
            templates.append(
                CandidateTemplate(
                    name=f"hybrid_mix_{mix:g}_eta_{eta:g}",
                    strategy="hybrid_load_imbalance",
                    family="hybrid_quadratic_variance",
                    eta=eta,
                    mix=mix,
                )
            )

    for p in (2.0, 3.0):
        for eta in (0.0, 0.5, 1.0):
            templates.append(
                CandidateTemplate(
                    name=f"total_load_p_{p:g}_eta_{eta:g}",
                    strategy="fluid_total_load",
                    family="total_load_power",
                    eta=eta,
                    p=p,
                )
            )

    return templates


def default_policy_points(alpha: float) -> list[PolicyPoint]:
    points = [
        PolicyPoint("uas_special", 1.0, 1.0, 1.0, alpha, "uas_special_case"),
        PolicyPoint("default", 0.85, 0.5, 0.5, alpha, "benchmark_default"),
        PolicyPoint("grid_b0p5_g0p25_c0p25", 0.5, 0.25, 0.25, alpha, "grid_candidate"),
        PolicyPoint("grid_b0p5_g0p5_c0p25", 0.5, 0.5, 0.25, alpha, "grid_candidate"),
        PolicyPoint("grid_b0p7_g0p25_c0p25", 0.7, 0.25, 0.25, alpha, "grid_candidate"),
    ]
    return points


def coarse_grid_policy_points(alpha: float) -> list[PolicyPoint]:
    points: list[PolicyPoint] = []
    for beta in (0.5, 0.7, 0.85, 1.0):
        for gamma in (0.25, 0.5, 0.75, 1.0):
            for c in (0.25, 0.5, 0.75, 1.0):
                points.append(
                    PolicyPoint(
                        name=f"b{beta:g}_g{gamma:g}_c{c:g}",
                        beta=beta,
                        gamma=gamma,
                        c=c,
                        alpha=alpha,
                        family="coarse_grid64",
                    )
                )
    return points


def axis_states(num_servers: int, magnitudes: Sequence[int]) -> np.ndarray:
    rows = []
    for i in range(num_servers):
        for mag in magnitudes:
            state = np.zeros(num_servers, dtype=np.int64)
            state[i] = int(mag)
            rows.append(state)
    return np.asarray(rows, dtype=np.int64)


def paired_imbalance_states(num_servers: int, magnitudes: Sequence[int]) -> np.ndarray:
    rows = []
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


def shell_random_states(
    *,
    num_servers: int,
    norms: Sequence[int],
    per_shell: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    probs = np.full(num_servers, 1.0 / num_servers, dtype=np.float64)
    for norm in norms:
        for _ in range(per_shell):
            rows.append(rng.multinomial(int(norm), probs))
    return np.asarray(rows, dtype=np.int64)


def trajectory_states(
    *,
    cfg: ExperimentConfig,
    reps: int,
    sim_time: float,
    sample_interval: float,
    seed: int,
    policy_point: PolicyPoint,
) -> np.ndarray:
    mu = np.asarray(cfg.system.service_rates, dtype=np.float64)
    policy = CalibratedUASRouting(
        mu=mu,
        alpha=policy_point.alpha,
        beta=policy_point.beta,
        gamma=policy_point.gamma,
        c=policy_point.c,
    )
    results = run_replications(
        num_servers=cfg.system.num_servers,
        arrival_rate=cfg.system.arrival_rate,
        service_rates=mu,
        policy=policy,
        num_replications=reps,
        sim_time=sim_time,
        sample_interval=sample_interval,
        base_seed=seed,
        progress_desc="proof-search trajectories",
    )
    return np.vstack([res.states for res in results]).astype(np.int64)


def unique_states(states: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.vstack([np.asarray(chunk, dtype=np.int64) for chunk in states if chunk.size > 0])
    if stacked.size == 0:
        return np.zeros((0, 0), dtype=np.int64)
    return np.unique(stacked, axis=0)


def build_state_bank(
    *,
    cfg: ExperimentConfig,
    seed: int,
    include_trajectories: bool,
    trajectory_reps: int,
    trajectory_sim_time: float,
    trajectory_sample_interval: float,
    trajectory_policy: PolicyPoint,
) -> np.ndarray:
    n = cfg.system.num_servers
    chunks = [
        np.zeros((1, n), dtype=np.int64),
        axis_states(n, magnitudes=(1, 2, 5, 10, 20, 40, 80, 160)),
        paired_imbalance_states(n, magnitudes=(4, 8, 16, 32, 64, 128)),
        shell_random_states(num_servers=n, norms=(5, 10, 20, 40, 80, 160), per_shell=64, seed=seed),
    ]
    if include_trajectories:
        chunks.append(
            trajectory_states(
                cfg=cfg,
                reps=trajectory_reps,
                sim_time=trajectory_sim_time,
                sample_interval=trajectory_sample_interval,
                seed=seed,
                policy_point=trajectory_policy,
            )
        )
    return unique_states(chunks)


def evaluate_template_on_policy(
    *,
    states: np.ndarray,
    mu: np.ndarray,
    arrival_rate: float,
    policy_point: PolicyPoint,
    template: CandidateTemplate,
    tail_norm_threshold: float,
) -> TemplateEvaluation:
    drifts = exact_generator_drift(
        states,
        mu=mu,
        arrival_rate=arrival_rate,
        policy_point=policy_point,
        template=template,
    )
    return TemplateEvaluation(
        row=summarize_drifts(
            states=states,
            drifts=drifts,
            policy_point=policy_point,
            template_name=template.name,
            template_strategy=template.strategy,
            template_family=template.family,
            eta=template.eta,
            p=template.p,
            theta=template.theta,
            mix=template.mix,
            tail_norm_threshold=tail_norm_threshold,
        ),
        drifts=drifts,
    )


def sampled_eventual_threshold(
    *,
    norms: np.ndarray,
    drifts: np.ndarray,
) -> float | None:
    unique_norms = np.unique(norms)
    for threshold in np.sort(unique_norms):
        mask = norms >= threshold
        if mask.any() and np.all(drifts[mask] < 0.0):
            return float(threshold)
    return None


def summarize_drifts(
    *,
    states: np.ndarray,
    drifts: np.ndarray,
    policy_point: PolicyPoint,
    template_name: str,
    template_strategy: str,
    template_family: str,
    eta: float,
    p: float | None,
    theta: float | None,
    mix: float | None,
    tail_norm_threshold: float,
    combo_support: str | None = None,
) -> dict[str, object]:
    norms = states.sum(axis=1).astype(np.float64)
    tail_mask = norms >= tail_norm_threshold
    tail_drifts = drifts[tail_mask]
    if tail_drifts.size == 0:
        tail_drifts = drifts
        tail_mask = np.ones_like(norms, dtype=bool)

    positive_tail = int(np.count_nonzero(tail_drifts > 0.0))
    eventual_threshold = sampled_eventual_threshold(norms=norms, drifts=drifts)
    return {
        "policy_name": policy_point.name,
        "policy_family": policy_point.family,
        "alpha": policy_point.alpha,
        "beta": policy_point.beta,
        "gamma": policy_point.gamma,
        "c": policy_point.c,
        "template_name": template_name,
        "template_strategy": template_strategy,
        "template_family": template_family,
        "eta": eta,
        "p": p,
        "theta": theta,
        "mix": mix,
        "combo_support": combo_support,
        "num_states": int(states.shape[0]),
        "tail_norm_threshold": float(tail_norm_threshold),
        "num_tail_states": int(tail_mask.sum()),
        "max_drift": float(drifts.max()),
        "mean_drift": float(drifts.mean()),
        "min_drift": float(drifts.min()),
        "max_tail_drift": float(tail_drifts.max()),
        "mean_tail_drift": float(tail_drifts.mean()),
        "positive_tail_fraction": float(positive_tail / tail_drifts.size),
        "negative_tail_fraction": float(np.count_nonzero(tail_drifts < 0.0) / tail_drifts.size),
        "positive_tail_count": positive_tail,
        "max_normalized_tail_drift": float(np.max(tail_drifts / np.maximum(norms[tail_mask], 1.0))),
        "passes_tail_sample": bool(np.all(tail_drifts < 0.0)),
        "sampled_eventual_threshold": eventual_threshold,
        "passes_eventual_tail_sample": bool(eventual_threshold is not None),
    }


def optimize_convex_template_combo(
    *,
    states: np.ndarray,
    mu: np.ndarray,
    arrival_rate: float,
    policy_point: PolicyPoint,
    template_evaluations: Sequence[TemplateEvaluation],
    tail_norm_threshold: float,
    scope_name: str,
) -> TemplateEvaluation | None:
    if not template_evaluations:
        return None

    norms = states.sum(axis=1).astype(np.float64)
    if scope_name == "tail":
        mask = norms >= tail_norm_threshold
        if not np.any(mask):
            mask = np.ones(states.shape[0], dtype=bool)
    elif scope_name == "all":
        mask = np.ones(states.shape[0], dtype=bool)
    else:
        raise ValueError(f"Unknown convex-combo scope: {scope_name}")

    basis_drifts = np.column_stack([evaluation.drifts for evaluation in template_evaluations])
    constrained_drifts = basis_drifts[mask]
    num_templates = constrained_drifts.shape[1]
    objective = np.zeros(num_templates + 1, dtype=np.float64)
    objective[-1] = 1.0

    a_ub = np.hstack([constrained_drifts, -np.ones((constrained_drifts.shape[0], 1), dtype=np.float64)])
    b_ub = np.zeros(constrained_drifts.shape[0], dtype=np.float64)
    a_eq = np.zeros((1, num_templates + 1), dtype=np.float64)
    a_eq[0, :num_templates] = 1.0
    b_eq = np.array([1.0], dtype=np.float64)
    bounds = [(0.0, None)] * num_templates + [(None, None)]

    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        log.warning("Convex combination LP failed for %s on %s: %s", policy_point.name, scope_name, result.message)
        return None

    weights = np.asarray(result.x[:num_templates], dtype=np.float64)
    combo_drifts = basis_drifts @ weights
    support_parts: list[str] = []
    for weight, evaluation in sorted(
        zip(weights, template_evaluations, strict=True),
        key=lambda item: float(item[0]),
        reverse=True,
    ):
        if weight > 1e-6:
            support_parts.append(f"{evaluation.row['template_name']}:{weight:.6f}")
    combo_name = f"convex_combo_lp_{scope_name}"
    row = summarize_drifts(
        states=states,
        drifts=combo_drifts,
        policy_point=policy_point,
        template_name=combo_name,
        template_strategy="convex_basis_combo",
        template_family="convex_combo",
        eta=float("nan"),
        p=None,
        theta=None,
        mix=None,
        tail_norm_threshold=tail_norm_threshold,
        combo_support="; ".join(support_parts[:12]),
    )
    row["lp_objective_max_drift"] = float(result.fun)
    row["lp_scope"] = scope_name
    row["lp_num_active_weights"] = int(np.count_nonzero(weights > 1e-6))
    return TemplateEvaluation(row=row, drifts=combo_drifts)


def rank_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    def _key(row: dict[str, object]) -> tuple[float, float, float]:
        return (
            float(row["max_tail_drift"]),
            float(row["positive_tail_fraction"]),
            float(row["mean_tail_drift"]),
        )

    return sorted(rows, key=_key)


def write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_best_rows(
    rows: Sequence[dict[str, object]],
    *,
    top_k: int,
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    by_policy: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy_name"]), []).append(row)
    for policy_name, policy_rows in by_policy.items():
        grouped[policy_name] = rank_rows(policy_rows)[:top_k]
    return grouped


def render_summary_markdown(
    *,
    cfg: ExperimentConfig,
    states: np.ndarray,
    policies: Sequence[PolicyPoint],
    templates: Sequence[CandidateTemplate],
    best_rows: dict[str, list[dict[str, object]]],
    tail_norm_threshold: float,
) -> str:
    lines = [
        "# Calibrated UAS Proof Search",
        "",
        "This report ranks numerically promising Lyapunov/proof templates.",
        "A passing score here is evidence for a direction, not a formal proof.",
        "",
        "## Benchmark",
        f"- num_servers: {cfg.system.num_servers}",
        f"- arrival_rate: {cfg.system.arrival_rate}",
        f"- service_rates: {list(cfg.system.service_rates)}",
        f"- sampled_states: {states.shape[0]}",
        f"- tail_norm_threshold: {tail_norm_threshold}",
        f"- policy_points: {len(policies)}",
        f"- proof_templates: {len(templates)}",
    ]
    for policy_name, rows in best_rows.items():
        lines.extend(["", f"## Best Templates For `{policy_name}`"])
        for row in rows:
            lines.append(
                "- "
                f"{row['template_name']} "
                f"[strategy={row['template_strategy']}] "
                f"max_tail_drift={row['max_tail_drift']:.6f}, "
                f"positive_tail_fraction={row['positive_tail_fraction']:.4f}, "
                f"mean_tail_drift={row['mean_tail_drift']:.6f}, "
                f"passes_tail_sample={row['passes_tail_sample']}, "
                f"sampled_eventual_threshold={row.get('sampled_eventual_threshold')}"
            )
            combo_support = row.get("combo_support")
            if combo_support:
                lines.append(f"  support: {combo_support}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search for numerically promising proof templates for the full Calibrated UAS family.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=20.0, help="Calibrated-policy alpha used in the search.")
    parser.add_argument(
        "--policy-grid",
        choices=["audit", "coarse64"],
        default="coarse64",
        help="Which set of calibrated parameter points to scan.",
    )
    parser.add_argument("--tail-norm-threshold", type=float, default=DEFAULT_TAIL_NORM)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-trajectory-states", action="store_true")
    parser.add_argument("--trajectory-reps", type=int, default=DEFAULT_TRAJ_REPS)
    parser.add_argument("--trajectory-sim-time", type=float, default=DEFAULT_TRAJ_SIM_TIME)
    parser.add_argument("--trajectory-sample-interval", type=float, default=DEFAULT_TRAJ_SAMPLE_INTERVAL)
    parser.add_argument("overrides", nargs="*", help="Optional OmegaConf dotlist overrides.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    cfg, resolved_raw = load_policy_cfg(
        config_name=args.config_name,
        overrides=args.overrides,
        output_dir=args.output_dir,
    )
    run_dir, run_id = get_run_config(cfg, "proof_search", resolved_raw)
    log.info("Calibrated UAS proof-search capsule: %s", run_dir)
    log.info("Run id: %s", run_id)

    policies = (
        default_policy_points(args.alpha)
        if args.policy_grid == "audit"
        else coarse_grid_policy_points(args.alpha)
    )
    templates = enumerate_templates()
    mu = np.asarray(cfg.system.service_rates, dtype=np.float64)

    trajectory_policy = PolicyPoint("trajectory_default", 0.85, 0.5, 0.5, args.alpha, "trajectory_seed")
    states = build_state_bank(
        cfg=cfg,
        seed=args.seed,
        include_trajectories=not args.no_trajectory_states,
        trajectory_reps=args.trajectory_reps,
        trajectory_sim_time=args.trajectory_sim_time,
        trajectory_sample_interval=args.trajectory_sample_interval,
        trajectory_policy=trajectory_policy,
    )
    log.info("State bank size: %d", states.shape[0])
    log.info("Policy points: %d", len(policies))
    log.info("Proof templates: %d", len(templates))

    rows: list[dict[str, object]] = []
    for policy in policies:
        log.info(
            "Evaluating policy point %s (beta=%.3f, gamma=%.3f, c=%.3f)",
            policy.name,
            policy.beta,
            policy.gamma,
            policy.c,
        )
        template_evaluations: list[TemplateEvaluation] = []
        for template in templates:
            evaluation = evaluate_template_on_policy(
                states=states,
                mu=mu,
                arrival_rate=cfg.system.arrival_rate,
                policy_point=policy,
                template=template,
                tail_norm_threshold=args.tail_norm_threshold,
            )
            template_evaluations.append(evaluation)
            rows.append(evaluation.row)

        for scope_name in ("tail", "all"):
            combo = optimize_convex_template_combo(
                states=states,
                mu=mu,
                arrival_rate=cfg.system.arrival_rate,
                policy_point=policy,
                template_evaluations=template_evaluations,
                tail_norm_threshold=args.tail_norm_threshold,
                scope_name=scope_name,
            )
            if combo is not None:
                rows.append(combo.row)

    ranked_rows = rank_rows(rows)
    best_rows = summarize_best_rows(ranked_rows, top_k=args.top_k)

    metrics_csv = metrics_path(run_dir, "proof_search_rankings.csv")
    write_csv(ranked_rows, metrics_csv)
    metadata_path(run_dir, "proof_search_rankings.json").write_text(
        json.dumps(ranked_rows, indent=2),
        encoding="utf-8",
    )
    metadata_path(run_dir, "proof_search_summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "policy_grid": args.policy_grid,
                "state_count": int(states.shape[0]),
                "tail_norm_threshold": float(args.tail_norm_threshold),
                "top_k": int(args.top_k),
                "best_rows": best_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata_path(run_dir, "proof_search_summary.md").write_text(
        render_summary_markdown(
            cfg=cfg,
            states=states,
            policies=policies,
            templates=templates,
            best_rows=best_rows,
            tail_norm_threshold=args.tail_norm_threshold,
        ),
        encoding="utf-8",
    )
    log.info("Top overall rows:")
    for row in ranked_rows[: min(args.top_k, len(ranked_rows))]:
        log.info(
            "  policy=%s template=%s strategy=%s max_tail_drift=%.6f positive_tail_fraction=%.4f",
            row["policy_name"],
            row["template_name"],
            row["template_strategy"],
            row["max_tail_drift"],
            row["positive_tail_fraction"],
        )
    log.info("Proof-search outputs written under %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
