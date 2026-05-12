"""
Probe the state-dependent-temperature UAS direction under the anchor benchmark.

This experiment does not rerun the published baselines. It reuses the saved
policy-comparison metrics for JSSQ / UAS / Calibrated UAS, evaluates only new
state-dependent UAS candidates, and reports whether the new direction clears
those fixed reference values.

The math-facing audit here is intentionally narrow: for sampled states, it
checks that the exact weighted-UAS generator drift of the implemented adaptive
policy remains below the same theorem-level bound

    LV(Q) <= -eps * |Q|_1 + R

used by the fixed-temperature UAS proof. This is an implementation sanity
check, not a symbolic proof.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from gibbsq.analysis.metrics import (
    gini_coefficient,
    sojourn_time_estimate,
    time_averaged_queue_lengths,
)
from gibbsq.core.config import (
    ExperimentConfig,
    drift_constant_R,
    drift_rate_epsilon,
    load_experiment_config,
)
from gibbsq.core.policies import StateDependentUASRouting
from gibbsq.engines.numpy_engine import run_replications
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.progress import iter_progress
from gibbsq.utils.run_artifacts import artifacts_dir, metrics_path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_POLICY_NAMES = (
    "JSSQ (Min Sojourn)",
    "UAS",
    "Calibrated UAS",
)


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    label: str
    params: dict[str, float]


def _select_probe_value(raw_cfg: DictConfig, key: str, default):
    value = OmegaConf.select(raw_cfg, f"probe.{key}", default=default)
    return default if value is None else value


def _build_piecewise_mean_potential_policy(
    mu: np.ndarray,
    *,
    t1: float,
    t2: float,
    a_low: float,
    a_mid: float,
    a_high: float,
    label: str,
) -> StateDependentUASRouting:
    def alpha_fn(Q: np.ndarray, mu_vec: np.ndarray) -> float:
        mean_potential = float(np.mean((Q + 1.0) / mu_vec))
        if mean_potential <= t1:
            return a_low
        if mean_potential <= t2:
            return a_mid
        return a_high

    return StateDependentUASRouting(mu, alpha_fn, label=label)


def _build_piecewise_cv_policy(
    mu: np.ndarray,
    *,
    threshold: float,
    alpha_balanced: float,
    alpha_imbalanced: float,
    label: str,
) -> StateDependentUASRouting:
    def alpha_fn(Q: np.ndarray, mu_vec: np.ndarray) -> float:
        potentials = (Q + 1.0) / mu_vec
        mean_potential = float(np.mean(potentials))
        cv = float(np.std(potentials) / max(mean_potential, 1e-8))
        if cv <= threshold:
            return alpha_balanced
        return alpha_imbalanced

    return StateDependentUASRouting(mu, alpha_fn, label=label)


def _candidate_policy(spec: CandidateSpec, mu: np.ndarray) -> StateDependentUASRouting:
    if spec.family == "mean_potential_3level":
        return _build_piecewise_mean_potential_policy(
            mu,
            t1=float(spec.params["t1"]),
            t2=float(spec.params["t2"]),
            a_low=float(spec.params["a_low"]),
            a_mid=float(spec.params["a_mid"]),
            a_high=float(spec.params["a_high"]),
            label=spec.label,
        )
    if spec.family == "cv_2level":
        return _build_piecewise_cv_policy(
            mu,
            threshold=float(spec.params["threshold"]),
            alpha_balanced=float(spec.params["alpha_balanced"]),
            alpha_imbalanced=float(spec.params["alpha_imbalanced"]),
            label=spec.label,
        )
    raise ValueError(f"Unsupported candidate family: {spec.family}")


def _generate_candidates(search_mode: str) -> list[CandidateSpec]:
    search_mode = str(search_mode).lower()
    if search_mode not in {"quick", "full"}:
        raise ValueError(f"search_mode must be 'quick' or 'full', got {search_mode!r}")

    threshold_pairs = [(1.5, 3.0), (2.0, 4.0)]
    alpha_triplets = [
        (10.0, 50.0, 100.0),
        (5.0, 20.0, 100.0),
        (100.0, 50.0, 10.0),
        (50.0, 20.0, 5.0),
    ]
    cv_thresholds = [0.20, 0.35, 0.50]
    alpha_pairs = [
        (100.0, 20.0),
        (50.0, 10.0),
        (20.0, 100.0),
        (10.0, 50.0),
    ]

    if search_mode == "full":
        threshold_pairs.extend([(2.5, 5.0), (3.0, 6.0)])
        alpha_triplets.extend([(20.0, 50.0, 100.0), (10.0, 20.0, 50.0)])
        cv_thresholds.extend([0.15, 0.65])
        alpha_pairs.extend([(100.0, 5.0), (5.0, 100.0)])

    candidates: list[CandidateSpec] = []
    for t1, t2 in threshold_pairs:
        for a_low, a_mid, a_high in alpha_triplets:
            label = (
                f"mp3_t({t1:.1f},{t2:.1f})_a({a_low:.0f},{a_mid:.0f},{a_high:.0f})"
            )
            candidates.append(
                CandidateSpec(
                    family="mean_potential_3level",
                    label=label,
                    params={
                        "t1": t1,
                        "t2": t2,
                        "a_low": a_low,
                        "a_mid": a_mid,
                        "a_high": a_high,
                    },
                )
            )

    for threshold in cv_thresholds:
        for alpha_balanced, alpha_imbalanced in alpha_pairs:
            label = (
                f"cv2_thr({threshold:.2f})_a({alpha_balanced:.0f},{alpha_imbalanced:.0f})"
            )
            candidates.append(
                CandidateSpec(
                    family="cv_2level",
                    label=label,
                    params={
                        "threshold": threshold,
                        "alpha_balanced": alpha_balanced,
                        "alpha_imbalanced": alpha_imbalanced,
                    },
                )
            )

    return candidates


def _evaluate_policy(
    policy: StateDependentUASRouting,
    cfg: ExperimentConfig,
    *,
    num_replications: int,
    sim_time: float,
    sample_interval: float,
    base_seed: int,
) -> dict[str, float]:
    results = run_replications(
        num_servers=cfg.system.num_servers,
        arrival_rate=cfg.system.arrival_rate,
        service_rates=np.asarray(cfg.system.service_rates, dtype=np.float64),
        policy=policy,
        num_replications=num_replications,
        sim_time=sim_time,
        sample_interval=sample_interval,
        base_seed=base_seed,
        progress_desc=policy.label,
    )

    burn_in = cfg.simulation.burn_in_fraction
    q_totals = [float(time_averaged_queue_lengths(r, burn_in).sum()) for r in results]
    ginis = [float(gini_coefficient(time_averaged_queue_lengths(r, burn_in))) for r in results]
    sojourns = [
        float(sojourn_time_estimate(r, cfg.system.arrival_rate, burn_in)) for r in results
    ]

    return {
        "mean_q_total": float(np.mean(q_totals)),
        "se_q_total": _standard_error(q_totals),
        "mean_gini": float(np.mean(ginis)),
        "mean_sojourn": float(np.mean(sojourns)),
    }


def _standard_error(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _load_reference_metrics() -> tuple[dict[str, float], Path | None]:
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
    reference_values: dict[str, float] = {}
    with newest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            name = row.get("policy")
            if name in REFERENCE_POLICY_NAMES:
                reference_values[name] = float(row["mean_q_total"])

    return reference_values, newest


def _adaptive_generator_drift(
    Q: np.ndarray,
    lam: float,
    mu: np.ndarray,
    policy: StateDependentUASRouting,
) -> float:
    probs = np.asarray(policy(Q, np.random.default_rng(0)), dtype=np.float64)
    q_float = Q.astype(np.float64)
    return float(
        lam * np.sum(probs * ((q_float + 0.5) / mu))
        - np.sum(q_float)
        + 0.5 * np.sum(q_float > 0)
    )


def _sample_audit_states(
    num_servers: int,
    *,
    seed: int,
    random_count: int,
    max_q: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = [np.zeros(num_servers, dtype=np.int64)]

    for total in (5, 10, 20, 40, 80):
        balanced = np.full(num_servers, total // num_servers, dtype=np.int64)
        balanced[: total % num_servers] += 1
        states.append(balanced)
        spike = np.zeros(num_servers, dtype=np.int64)
        spike[0] = total
        states.append(spike)

    for _ in range(random_count):
        states.append(rng.integers(0, max_q + 1, size=num_servers, dtype=np.int64))

    return states


def _audit_drift_bound(
    policy: StateDependentUASRouting,
    cfg: ExperimentConfig,
    *,
    sample_count: int,
    max_q: int,
    seed: int,
) -> dict[str, float]:
    mu = np.asarray(cfg.system.service_rates, dtype=np.float64)
    lam = float(cfg.system.arrival_rate)
    epsilon = float(drift_rate_epsilon(cfg))
    R = float(drift_constant_R(cfg))

    states = _sample_audit_states(
        cfg.system.num_servers,
        seed=seed,
        random_count=sample_count,
        max_q=max_q,
    )

    max_violation = float("-inf")
    worst_q_norm = 0.0
    for Q in states:
        exact = _adaptive_generator_drift(Q, lam, mu, policy)
        bound = -epsilon * float(np.sum(Q)) + R
        violation = exact - bound
        if violation > max_violation:
            max_violation = float(violation)
            worst_q_norm = float(np.sum(Q))

    return {
        "epsilon": epsilon,
        "R": R,
        "max_violation": max_violation,
        "worst_q_norm": worst_q_norm,
        "sampled_states": float(len(states)),
    }


def _write_summary(
    run_dir: Path,
    *,
    references: dict[str, float],
    reference_path: Path | None,
    finalists: list[dict[str, object]],
    recommendation: str,
) -> None:
    summary_path = artifacts_dir(run_dir) / "probe_summary.md"
    lines = [
        "# State-Dependent UAS Probe",
        "",
        f"- Reference metrics source: `{reference_path}`" if reference_path else "- Reference metrics source: not found",
        f"- JSSQ reference: `{references.get('JSSQ (Min Sojourn)', float('nan')):.6f}`" if references else "- JSSQ reference: unavailable",
        f"- UAS reference: `{references.get('UAS', float('nan')):.6f}`" if references else "- UAS reference: unavailable",
        f"- Calibrated UAS reference: `{references.get('Calibrated UAS', float('nan')):.6f}`" if references else "- Calibrated UAS reference: unavailable",
        "",
        "## Finalists",
        "",
    ]
    for row in finalists:
        drift = row["drift_audit"]
        lines.append(
            f"- `{row['label']}`: mean_q_total=`{row['mean_q_total']:.6f}`, "
            f"se=`{row['se_q_total']:.6f}`, max_violation=`{drift['max_violation']:.3e}`"
        )
    lines.extend(["", "## Recommendation", "", recommendation, ""])
    summary_path.write_text("\n".join(lines), encoding="utf-8")


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig) -> None:
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "policy")
    run_dir, run_id = get_run_config(cfg, "state_dependent_uas_probe", resolved_raw_cfg)
    run_logger = setup_wandb(
        cfg,
        resolved_raw_cfg,
        default_group="state_dependent_uas_probe",
        run_id=run_id,
        run_dir=run_dir,
    )

    search_mode = str(_select_probe_value(raw_cfg, "search_mode", "quick"))
    top_k = int(_select_probe_value(raw_cfg, "top_k", 3))
    pilot_replications = int(_select_probe_value(raw_cfg, "pilot_replications", 6))
    pilot_sim_time = float(_select_probe_value(raw_cfg, "pilot_sim_time", 4000.0))
    final_replications = int(
        _select_probe_value(raw_cfg, "final_replications", cfg.simulation.num_replications)
    )
    final_sim_time = float(
        _select_probe_value(raw_cfg, "final_sim_time", cfg.simulation.ssa.sim_time)
    )
    audit_sample_count = int(_select_probe_value(raw_cfg, "audit_sample_count", 2000))
    audit_max_q = int(_select_probe_value(raw_cfg, "audit_max_q", 40))

    references, reference_path = _load_reference_metrics()
    candidates = _generate_candidates(search_mode)
    mu = np.asarray(cfg.system.service_rates, dtype=np.float64)

    log.info("Loaded %d adaptive candidates for search mode '%s'.", len(candidates), search_mode)
    if references:
        log.info("Reference policy values: %s", references)
    else:
        log.warning("No saved reference policy metrics were found. The probe will report raw values only.")

    pilot_rows: list[dict[str, object]] = []
    for spec in iter_progress(
        candidates,
        total=len(candidates),
        desc="lt-uas pilot",
        unit="candidate",
        leave=False,
    ):
        policy = _candidate_policy(spec, mu)
        metrics = _evaluate_policy(
            policy,
            cfg,
            num_replications=pilot_replications,
            sim_time=pilot_sim_time,
            sample_interval=cfg.simulation.ssa.sample_interval,
            base_seed=cfg.simulation.seed,
        )
        row: dict[str, object] = {
            "phase": "pilot",
            "label": spec.label,
            "family": spec.family,
            "params": spec.params,
            **metrics,
        }
        pilot_rows.append(row)
        append_metrics_jsonl(row, metrics_path(run_dir, "pilot_metrics.jsonl"))

    pilot_rows.sort(key=lambda item: float(item["mean_q_total"]))
    finalists_specs = pilot_rows[:top_k]

    final_rows: list[dict[str, object]] = []
    for rank, finalist in enumerate(finalists_specs, start=1):
        spec = CandidateSpec(
            family=str(finalist["family"]),
            label=str(finalist["label"]),
            params=dict(finalist["params"]),
        )
        policy = _candidate_policy(spec, mu)
        metrics = _evaluate_policy(
            policy,
            cfg,
            num_replications=final_replications,
            sim_time=final_sim_time,
            sample_interval=cfg.simulation.ssa.sample_interval,
            base_seed=cfg.simulation.seed + 10_000 * rank,
        )
        drift_audit = _audit_drift_bound(
            policy,
            cfg,
            sample_count=audit_sample_count,
            max_q=audit_max_q,
            seed=cfg.simulation.seed + 50_000 * rank,
        )
        row = {
            "phase": "final",
            "rank": rank,
            "label": spec.label,
            "family": spec.family,
            "params": spec.params,
            **metrics,
            "drift_audit": drift_audit,
        }
        final_rows.append(row)
        append_metrics_jsonl(row, metrics_path(run_dir, "final_metrics.jsonl"))

    final_rows.sort(key=lambda item: float(item["mean_q_total"]))
    best = final_rows[0] if final_rows else None

    recommendation = "No finalists were produced."
    if best is not None:
        best_q = float(best["mean_q_total"])
        jssq = references.get("JSSQ (Min Sojourn)")
        cal_uas = references.get("Calibrated UAS")
        drift_ok = float(best["drift_audit"]["max_violation"]) <= 1e-9

        if drift_ok and cal_uas is not None and best_q < cal_uas:
            recommendation = (
                "Proceed: the best adaptive-temperature UAS candidate beat the saved "
                "Calibrated UAS benchmark while preserving the sampled theorem-style drift bound."
            )
        elif drift_ok and jssq is not None and best_q < jssq:
            recommendation = (
                "Proceed cautiously: the best adaptive-temperature UAS candidate beat "
                "JSSQ but did not clear the saved Calibrated UAS benchmark."
            )
        elif not drift_ok:
            recommendation = (
                "Do not pivot yet: the best adaptive implementation violated the sampled "
                "theorem-style drift bound, so the implementation or the claim needs correction first."
            )
        else:
            recommendation = (
                "Do not pivot yet: the adaptive candidates did not beat the saved reference "
                "targets strongly enough to justify changing the project direction."
            )

    summary_payload = {
        "references": references,
        "reference_path": str(reference_path) if reference_path else None,
        "pilot_top_k": finalists_specs,
        "finalists": final_rows,
        "recommendation": recommendation,
    }
    summary_json = artifacts_dir(run_dir) / "probe_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_summary(
        run_dir,
        references=references,
        reference_path=reference_path,
        finalists=final_rows,
        recommendation=recommendation,
    )

    if run_logger:
        run_logger.log({"probe/best_mean_q_total": best["mean_q_total"] if best else None})
        run_logger.finish()


if __name__ == "__main__":
    main()
