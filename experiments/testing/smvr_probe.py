"""
Final empirical probe for the professor's value-routing architecture.

This script evaluates a minimal proof-friendly instantiation of the SMVR idea:

    p_i(Q) proportional to mu_i**gamma * exp(-alpha * A_i(Q))

where the advantage surrogate A_i(Q) is the calibrated-UAS energy plus
non-negative convex value corrections. The family contains Calibrated UAS
exactly, so this probe answers the only decision question that matters:

    does any value-routing candidate beat Calibrated UAS on the anchor
    benchmark, under the same evaluation protocol?
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from gibbsq.analysis.metrics import (
    gini_coefficient,
    sojourn_time_estimate,
    time_averaged_queue_lengths,
)
from gibbsq.core.config import ExperimentConfig, load_experiment_config
from gibbsq.core.policies import (
    CalibratedUASRouting,
    JSSQRouting,
    QuadraticSMVRRouting,
    RoutingPolicy,
    UASRouting,
)
from gibbsq.engines.numpy_engine import run_replications
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.progress import iter_progress
from gibbsq.utils.run_artifacts import artifacts_dir, metrics_path

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _select_probe_value(raw_cfg: DictConfig, key: str, default):
    value = OmegaConf.select(raw_cfg, f"probe.{key}", default=default)
    return default if value is None else value


def _standard_error(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _ci95_halfwidth(values: np.ndarray) -> float:
    return 1.96 * _standard_error(values)


def _load_reference_metrics() -> tuple[dict[str, dict[str, float]], Path | None]:
    candidates = []
    for root in (
        PROJECT_ROOT / "outputs" / "final" / "policy",
        PROJECT_ROOT / "outputs_legacy" / "outputs" / "final" / "policy",
    ):
        if root.exists():
            candidates.extend(root.glob("*/metrics/corrected_comparison_metrics.jsonl"))

    if not candidates:
        return {}, None

    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    refs: dict[str, dict[str, float]] = {}
    with newest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            refs[str(row["policy"])] = {
                "mean_q_total": float(row["mean_q_total"]),
                "se_q_total": float(row.get("se_q_total", 0.0)),
                "mean_gini": float(row.get("mean_gini", float("nan"))),
                "mean_sojourn": float(row.get("mean_sojourn", float("nan"))),
            }
    return refs, newest


@dataclass(frozen=True)
class SMVRSpec:
    label: str
    alpha: float
    gamma: float
    local_strength: float
    total_strength: float
    beta: float
    c: float


def _smvr_candidates(search_mode: str, *, beta: float, c: float) -> list[SMVRSpec]:
    search_mode = str(search_mode).lower()
    if search_mode not in {"quick", "full"}:
        raise ValueError(f"search_mode must be 'quick' or 'full', got {search_mode!r}")

    alphas = [18.0, 20.0]
    gammas = [0.5, 1.0]
    local_strengths = [0.0, 0.20]
    total_strengths = [0.0, 0.05]

    if search_mode == "full":
        alphas = [15.0, 20.0, 25.0]
        local_strengths = [0.0, 0.10, 0.25]
        total_strengths = [0.0, 0.05]

    candidates: list[SMVRSpec] = []
    seen: set[tuple[float, float, float, float, float, float]] = set()
    for alpha in alphas:
        for gamma in gammas:
            for local_strength in local_strengths:
                for total_strength in total_strengths:
                    key = (alpha, gamma, local_strength, total_strength, beta, c)
                    if key in seen:
                        continue
                    seen.add(key)
                    label = (
                        f"smvr_a({alpha:g})_g({gamma:g})_"
                        f"l({local_strength:g})_t({total_strength:g})"
                    )
                    candidates.append(
                        SMVRSpec(
                            label=label,
                            alpha=float(alpha),
                            gamma=float(gamma),
                            local_strength=float(local_strength),
                            total_strength=float(total_strength),
                            beta=float(beta),
                            c=float(c),
                        )
                    )
    return candidates


def _representative_states(num_servers: int, *, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = [np.zeros(num_servers, dtype=np.int64)]
    for total in (5, 10, 20, 40, 80):
        states.append(np.full(num_servers, total // num_servers, dtype=np.int64))
        spike = np.zeros(num_servers, dtype=np.int64)
        spike[0] = total
        states.append(spike)
    for _ in range(12):
        states.append(rng.integers(0, 20, size=num_servers, dtype=np.int64))
    return states


def _policy_fingerprint(policy: RoutingPolicy, states: list[np.ndarray]) -> tuple[float, ...]:
    rng = np.random.default_rng(0)
    parts: list[float] = []
    for state in states:
        probs = np.asarray(policy(state, rng), dtype=np.float64)
        parts.extend(np.round(probs, 12).tolist())
    return tuple(parts)


def _uniqueness_audit(
    specs: list[SMVRSpec],
    mu: np.ndarray,
    *,
    seed: int,
) -> tuple[list[SMVRSpec], dict[str, object]]:
    states = _representative_states(mu.size, seed=seed)
    seen: dict[tuple[float, ...], str] = {}
    unique_specs: list[SMVRSpec] = []
    duplicates: list[dict[str, str]] = []

    for spec in specs:
        policy = QuadraticSMVRRouting(
            mu,
            alpha=spec.alpha,
            beta=spec.beta,
            gamma=spec.gamma,
            c=spec.c,
            local_strength=spec.local_strength,
            total_strength=spec.total_strength,
        )
        fingerprint = _policy_fingerprint(policy, states)
        original = seen.get(fingerprint)
        if original is None:
            seen[fingerprint] = spec.label
            unique_specs.append(spec)
        else:
            duplicates.append({"label": spec.label, "matches": original})

    audit = {
        "num_candidates": len(specs),
        "num_unique": len(unique_specs),
        "num_duplicates": len(duplicates),
        "duplicates": duplicates,
    }
    return unique_specs, audit


def _evaluate_policy(
    policy: RoutingPolicy,
    cfg: ExperimentConfig,
    *,
    num_replications: int,
    sim_time: float,
    sample_interval: float,
    base_seed: int,
    progress_desc: str,
) -> dict[str, object]:
    results = run_replications(
        num_servers=cfg.system.num_servers,
        arrival_rate=cfg.system.arrival_rate,
        service_rates=np.asarray(cfg.system.service_rates, dtype=np.float64),
        policy=policy,
        num_replications=num_replications,
        sim_time=sim_time,
        sample_interval=sample_interval,
        base_seed=base_seed,
        progress_desc=progress_desc,
    )

    burn_in = cfg.simulation.burn_in_fraction
    q_totals = np.asarray(
        [float(time_averaged_queue_lengths(r, burn_in).sum()) for r in results],
        dtype=np.float64,
    )
    ginis = np.asarray(
        [float(gini_coefficient(time_averaged_queue_lengths(r, burn_in))) for r in results],
        dtype=np.float64,
    )
    sojourns = np.asarray(
        [float(sojourn_time_estimate(r, cfg.system.arrival_rate, burn_in)) for r in results],
        dtype=np.float64,
    )
    return {
        "mean_q_total": float(np.mean(q_totals)),
        "se_q_total": _standard_error(q_totals),
        "mean_gini": float(np.mean(ginis)),
        "se_gini": _standard_error(ginis),
        "mean_sojourn": float(np.mean(sojourns)),
        "se_sojourn": _standard_error(sojourns),
        "q_totals": q_totals,
        "ginis": ginis,
        "sojourns": sojourns,
    }


def _paired_delta(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    delta = np.asarray(candidate - baseline, dtype=np.float64)
    mean = float(np.mean(delta))
    halfwidth = _ci95_halfwidth(delta)
    return {
        "mean_delta": mean,
        "ci95_low": mean - halfwidth,
        "ci95_high": mean + halfwidth,
        "se_delta": _standard_error(delta),
    }


def _write_summary(
    run_dir: Path,
    *,
    reference_path: Path | None,
    references: dict[str, dict[str, float]],
    audit: dict[str, object],
    baselines: list[dict[str, object]],
    finalists: list[dict[str, object]],
    paired_deltas: dict[str, dict[str, float]],
    recommendation: str,
) -> None:
    summary_path = artifacts_dir(run_dir) / "smvr_probe_summary.md"
    lines = [
        "# SMVR Probe",
        "",
        f"- Reference metrics source: `{reference_path}`" if reference_path else "- Reference metrics source: not found",
        f"- Saved Calibrated UAS: `{references.get('Calibrated UAS', {}).get('mean_q_total', float('nan')):.6f}`" if references else "- Saved Calibrated UAS: unavailable",
        "",
        "## Audit",
        "",
        f"- Candidates generated: `{audit['num_candidates']}`",
        f"- Behaviorally unique candidates: `{audit['num_unique']}`",
        f"- Duplicates removed: `{audit['num_duplicates']}`",
        "",
        "## Baselines",
        "",
    ]
    for row in baselines:
        lines.append(
            f"- `{row['label']}`: mean_q_total=`{row['mean_q_total']:.6f}`, "
            f"se=`{row['se_q_total']:.6f}`"
        )
    lines.extend(["", "## Finalists", ""])
    for row in finalists:
        lines.append(
            f"- `{row['label']}`: mean_q_total=`{row['mean_q_total']:.6f}`, "
            f"se=`{row['se_q_total']:.6f}`, mean_sojourn=`{row['mean_sojourn']:.6f}`"
        )
        delta = paired_deltas.get(str(row["label"]))
        if delta is not None:
            lines.append(
                f"delta vs Calibrated UAS for `{row['label']}`: `{delta['mean_delta']:+.6f}` "
                f"(95% CI `{delta['ci95_low']:+.6f}`, `{delta['ci95_high']:+.6f}`)"
            )
    lines.extend(["", "## Recommendation", "", recommendation, ""])
    summary_path.write_text("\n".join(lines), encoding="utf-8")


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig) -> None:
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "policy")
    run_dir, run_id = get_run_config(cfg, "smvr_probe", resolved_raw_cfg)
    run_logger = setup_wandb(
        cfg,
        resolved_raw_cfg,
        default_group="smvr_probe",
        run_id=run_id,
        run_dir=run_dir,
    )

    references, reference_path = _load_reference_metrics()
    mu = np.asarray(cfg.system.service_rates, dtype=np.float64)

    beta = float(_select_probe_value(raw_cfg, "smvr.beta", 0.85))
    c = float(_select_probe_value(raw_cfg, "smvr.c", 0.5))
    search_mode = str(_select_probe_value(raw_cfg, "smvr.search_mode", "quick"))
    top_k = int(_select_probe_value(raw_cfg, "smvr.top_k", 4))
    pilot_replications = int(_select_probe_value(raw_cfg, "smvr.pilot_replications", 4))
    pilot_sim_time = float(_select_probe_value(raw_cfg, "smvr.pilot_sim_time", 3000.0))
    final_replications = int(
        _select_probe_value(raw_cfg, "smvr.final_replications", cfg.simulation.num_replications)
    )
    final_sim_time = float(
        _select_probe_value(raw_cfg, "smvr.final_sim_time", cfg.simulation.ssa.sim_time)
    )

    candidates = _smvr_candidates(search_mode, beta=beta, c=c)
    max_candidates = int(_select_probe_value(raw_cfg, "smvr.max_candidates", 0))
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    unique_candidates, audit = _uniqueness_audit(
        candidates,
        mu,
        seed=int(_select_probe_value(raw_cfg, "smvr.audit_seed", cfg.simulation.seed)),
    )
    (artifacts_dir(run_dir) / "uniqueness_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    min_unique = int(_select_probe_value(raw_cfg, "smvr.min_unique_candidates", 8))
    if len(unique_candidates) < min_unique:
        raise RuntimeError(
            f"SMVR uniqueness audit kept only {len(unique_candidates)} candidates; "
            f"need at least {min_unique} to justify a long run."
        )

    pilot_rows: list[dict[str, object]] = []
    for spec in iter_progress(
        unique_candidates,
        total=len(unique_candidates),
        desc="smvr pilot",
        unit="candidate",
        leave=False,
    ):
        policy = QuadraticSMVRRouting(
            mu,
            alpha=spec.alpha,
            beta=spec.beta,
            gamma=spec.gamma,
            c=spec.c,
            local_strength=spec.local_strength,
            total_strength=spec.total_strength,
        )
        metrics = _evaluate_policy(
            policy,
            cfg,
            num_replications=pilot_replications,
            sim_time=pilot_sim_time,
            sample_interval=cfg.simulation.ssa.sample_interval,
            base_seed=cfg.simulation.seed,
            progress_desc=spec.label,
        )
        row = {
            "phase": "pilot",
            "label": spec.label,
            "alpha": spec.alpha,
            "gamma": spec.gamma,
            "local_strength": spec.local_strength,
            "total_strength": spec.total_strength,
            "beta": spec.beta,
            "c": spec.c,
            "mean_q_total": metrics["mean_q_total"],
            "se_q_total": metrics["se_q_total"],
            "mean_gini": metrics["mean_gini"],
            "mean_sojourn": metrics["mean_sojourn"],
        }
        pilot_rows.append(row)
        append_metrics_jsonl(row, metrics_path(run_dir, "pilot_metrics.jsonl"))

    pilot_rows.sort(key=lambda row: float(row["mean_q_total"]))
    finalists = pilot_rows[:top_k]

    baseline_specs = [
        ("JSSQ (Min Sojourn)", JSSQRouting(mu)),
        ("UAS", UASRouting(mu, alpha=10.0)),
        ("Calibrated UAS", CalibratedUASRouting(mu, alpha=20.0, beta=0.85, gamma=0.5, c=0.5)),
    ]
    baseline_rows: list[dict[str, object]] = []
    baseline_metrics: dict[str, dict[str, object]] = {}
    for label, policy in baseline_specs:
        metrics = _evaluate_policy(
            policy,
            cfg,
            num_replications=final_replications,
            sim_time=final_sim_time,
            sample_interval=cfg.simulation.ssa.sample_interval,
            base_seed=cfg.simulation.seed,
            progress_desc=label,
        )
        baseline_metrics[label] = metrics
        row = {
            "phase": "baseline",
            "label": label,
            "mean_q_total": metrics["mean_q_total"],
            "se_q_total": metrics["se_q_total"],
            "mean_gini": metrics["mean_gini"],
            "se_gini": metrics["se_gini"],
            "mean_sojourn": metrics["mean_sojourn"],
            "se_sojourn": metrics["se_sojourn"],
        }
        baseline_rows.append(row)
        append_metrics_jsonl(row, metrics_path(run_dir, "final_metrics.jsonl"))

    final_rows: list[dict[str, object]] = []
    paired_deltas: dict[str, dict[str, float]] = {}
    calibrated_q = np.asarray(baseline_metrics["Calibrated UAS"]["q_totals"], dtype=np.float64)
    jssq_q = np.asarray(baseline_metrics["JSSQ (Min Sojourn)"]["q_totals"], dtype=np.float64)
    uas_q = np.asarray(baseline_metrics["UAS"]["q_totals"], dtype=np.float64)

    for finalist in finalists:
        policy = QuadraticSMVRRouting(
            mu,
            alpha=float(finalist["alpha"]),
            beta=float(finalist["beta"]),
            gamma=float(finalist["gamma"]),
            c=float(finalist["c"]),
            local_strength=float(finalist["local_strength"]),
            total_strength=float(finalist["total_strength"]),
        )
        metrics = _evaluate_policy(
            policy,
            cfg,
            num_replications=final_replications,
            sim_time=final_sim_time,
            sample_interval=cfg.simulation.ssa.sample_interval,
            base_seed=cfg.simulation.seed,
            progress_desc=str(finalist["label"]),
        )
        q_totals = np.asarray(metrics["q_totals"], dtype=np.float64)
        row = {
            "phase": "finalist",
            "label": finalist["label"],
            "alpha": finalist["alpha"],
            "gamma": finalist["gamma"],
            "local_strength": finalist["local_strength"],
            "total_strength": finalist["total_strength"],
            "beta": finalist["beta"],
            "c": finalist["c"],
            "mean_q_total": metrics["mean_q_total"],
            "se_q_total": metrics["se_q_total"],
            "mean_gini": metrics["mean_gini"],
            "se_gini": metrics["se_gini"],
            "mean_sojourn": metrics["mean_sojourn"],
            "se_sojourn": metrics["se_sojourn"],
        }
        final_rows.append(row)
        append_metrics_jsonl(row, metrics_path(run_dir, "final_metrics.jsonl"))
        paired_deltas[str(finalist["label"])] = {
            "vs_calibrated": _paired_delta(q_totals, calibrated_q),
            "vs_jssq": _paired_delta(q_totals, jssq_q),
            "vs_uas": _paired_delta(q_totals, uas_q),
        }

    final_rows.sort(key=lambda row: float(row["mean_q_total"]))
    best = final_rows[0]
    best_delta = paired_deltas[str(best["label"])]["vs_calibrated"]
    has_value_correction = (
        float(best["local_strength"]) > 0.0 or float(best["total_strength"]) > 0.0
    )

    recommendation = (
        "Stop: the tested SMVR family did not beat Calibrated UAS. "
        "This direction is not worth promoting from probe to manuscript."
    )
    if best_delta["ci95_high"] < 0.0 and has_value_correction:
        recommendation = (
            "Proceed: the best SMVR candidate beat Calibrated UAS on matched seeds "
            "with a strictly negative 95% paired confidence interval."
        )
    elif best_delta["ci95_high"] < 0.0 and not has_value_correction:
        recommendation = (
            "Stop: the only improvement came from a zero-correction candidate, which is "
            "just a re-tuned calibrated baseline rather than evidence for the new "
            "value-routing architecture."
        )

    paired_summary = {
        label: {
            "vs_calibrated": delta["vs_calibrated"],
            "vs_jssq": delta["vs_jssq"],
            "vs_uas": delta["vs_uas"],
        }
        for label, delta in paired_deltas.items()
    }
    (artifacts_dir(run_dir) / "paired_deltas.json").write_text(
        json.dumps(paired_summary, indent=2),
        encoding="utf-8",
    )

    summary_payload = {
        "reference_path": str(reference_path) if reference_path else None,
        "references": references,
        "audit": audit,
        "baselines": baseline_rows,
        "finalists": final_rows,
        "paired_deltas": paired_summary,
        "recommendation": recommendation,
    }
    (artifacts_dir(run_dir) / "smvr_probe_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    _write_summary(
        run_dir,
        reference_path=reference_path,
        references=references,
        audit=audit,
        baselines=baseline_rows,
        finalists=final_rows,
        paired_deltas={label: delta["vs_calibrated"] for label, delta in paired_deltas.items()},
        recommendation=recommendation,
    )

    if run_logger:
        run_logger.log({"smvr/best_mean_q_total": best["mean_q_total"]})
        run_logger.finish()


if __name__ == "__main__":
    main()
