"""
Engine parity verification for publication-critical closed-form baselines.

This runner checks whether NumPy SSA and JAX SSA produce materially
equivalent results for the publication-relevant closed-form policies
under the exact scenario contracts used by the paper-facing experiments.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy import stats

from gibbsq.analysis.metrics import time_averaged_queue_lengths
from gibbsq.core.builders import build_policy_by_name
from gibbsq.core.config import (
    ExperimentConfig,
    critical_load_sim_time,
    hydra_to_config,
    load_experiment_config,
    resolve_experiment_config,
    validate,
)
from gibbsq.engines.jax_engine import compute_configured_max_events, policy_name_to_type, run_replications_jax
from gibbsq.engines.numpy_engine import SimResult, run_replications
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.logging import get_run_config, setup_wandb
from gibbsq.utils.run_artifacts import figure_path, metadata_path, metrics_path

log = logging.getLogger(__name__)
matplotlib.use("Agg")


@dataclass(frozen=True)
class ScenarioSpec:
    contract: str
    name: str
    policy_name: str
    label: str
    alpha: float
    num_servers: int
    arrival_rate: float
    service_rates: list[float]
    sim_time: float
    sample_interval: float
    burn_in_fraction: float
    num_replications: int
    base_seed: int


@dataclass(frozen=True)
class EngineMetrics:
    mean_q_total: float
    std_q_total: float
    se_q_total: float
    mean_final_q_total: float
    mean_arrival_rate: float
    mean_departure_rate: float
    mean_q_vector: list[float]
    per_rep_q_total: list[float]
    truncated: bool


def _policy_label(name: str) -> str:
    mapping = {
        "jsq": "JSQ",
        "jssq": "JSSQ",
        "uas": "UAS",
        "calibrated_uas": "Calibrated UAS",
        "refined_uas": "Calibrated UAS",
    }
    return mapping.get(name, name)


def _policy_contract_alpha(policy_name: str, default_alpha: float) -> float:
    if policy_name == "uas":
        return 10.0
    if policy_name in {"calibrated_uas", "refined_uas"}:
        return 20.0
    return float(default_alpha)


def _replication_budget(cfg: ExperimentConfig) -> int:
    ep = cfg.engine_parity
    if ep.mode == "full":
        return min(int(cfg.simulation.num_replications), int(ep.full_num_replications))
    return min(int(cfg.simulation.num_replications), int(ep.quick_num_replications))


def _generalize_cells(scale_vals: list[float], rho_vals: list[float], max_cells: int) -> list[tuple[float, float]]:
    candidates = [
        (scale_vals[0], rho_vals[0]),
        (scale_vals[0], rho_vals[-1]),
        (scale_vals[-1], rho_vals[0]),
        (scale_vals[-1], rho_vals[-1]),
        (scale_vals[len(scale_vals) // 2], rho_vals[len(rho_vals) // 2]),
    ]
    cells: list[tuple[float, float]] = []
    for cell in candidates:
        if cell not in cells:
            cells.append(cell)
        if len(cells) >= max_cells:
            break
    return cells


def _critical_rhos(rho_vals: list[float], max_rhos: int) -> list[float]:
    if len(rho_vals) <= max_rhos:
        return rho_vals
    picks = [rho_vals[0], rho_vals[-1]]
    if max_rhos > 2:
        mid = rho_vals[len(rho_vals) // 2]
        if mid not in picks:
            picks.insert(1, mid)
    return picks[:max_rhos]


def _contract_cfg(raw_cfg: DictConfig, profile_name: str, experiment_name: str) -> ExperimentConfig:
    resolved = resolve_experiment_config(raw_cfg, experiment_name, profile_name=profile_name)
    cfg = hydra_to_config(resolved)
    validate(cfg)
    return cfg


def _build_scenarios(raw_cfg: DictConfig, profile_name: str, cfg: ExperimentConfig) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    ep = cfg.engine_parity

    if ep.include_policy_contract:
        policy_cfg = _contract_cfg(raw_cfg, profile_name, "policy")
        for policy_name in ep.policies:
            scenarios.append(
                ScenarioSpec(
                    contract="policy",
                    name=f"policy/{policy_name}",
                    policy_name=policy_name,
                    label=f"policy::{_policy_label(policy_name)}",
                    alpha=_policy_contract_alpha(policy_name, policy_cfg.system.alpha),
                    num_servers=policy_cfg.system.num_servers,
                    arrival_rate=float(policy_cfg.system.arrival_rate),
                    service_rates=list(policy_cfg.system.service_rates),
                    sim_time=float(policy_cfg.simulation.ssa.sim_time),
                    sample_interval=float(policy_cfg.simulation.ssa.sample_interval),
                    burn_in_fraction=float(policy_cfg.simulation.burn_in_fraction),
                    num_replications=_replication_budget(policy_cfg),
                    base_seed=int(policy_cfg.simulation.seed),
                )
            )

    if ep.include_stats_contract:
        stats_cfg = _contract_cfg(raw_cfg, profile_name, "stats")
        scenarios.append(
            ScenarioSpec(
                contract="stats",
                name="stats/calibrated_uas",
                policy_name="calibrated_uas",
                label="stats::Calibrated UAS",
                alpha=float(stats_cfg.system.alpha),
                num_servers=stats_cfg.system.num_servers,
                arrival_rate=float(stats_cfg.system.arrival_rate),
                service_rates=list(stats_cfg.system.service_rates),
                sim_time=float(stats_cfg.simulation.ssa.sim_time),
                sample_interval=float(stats_cfg.simulation.ssa.sample_interval),
                burn_in_fraction=float(stats_cfg.simulation.burn_in_fraction),
                num_replications=_replication_budget(stats_cfg),
                base_seed=int(stats_cfg.simulation.seed),
            )
        )

    if ep.include_generalize_contract:
        gen_cfg = _contract_cfg(raw_cfg, profile_name, "generalize")
        scales = list(gen_cfg.generalization.scale_vals)
        rhos = list(gen_cfg.generalization.rho_grid_vals)
        if ep.mode == "full":
            cells = [(scale, rho) for scale in scales for rho in rhos]
        else:
            cells = _generalize_cells(scales, rhos, int(ep.quick_generalize_max_cells))
        for idx, (scale, rho) in enumerate(cells):
            service_rates = (np.asarray(gen_cfg.system.service_rates, dtype=np.float64) * float(scale)).tolist()
            arrival_rate = float(rho * np.sum(service_rates))
            scenarios.append(
                ScenarioSpec(
                    contract="generalize",
                    name=f"generalize/scale={scale}/rho={rho}",
                    policy_name="calibrated_uas",
                    label=f"generalize::Calibrated UAS::{scale}x::{rho:.2f}",
                    alpha=float(gen_cfg.system.alpha),
                    num_servers=gen_cfg.system.num_servers,
                    arrival_rate=arrival_rate,
                    service_rates=service_rates,
                    sim_time=float(gen_cfg.simulation.ssa.sim_time),
                    sample_interval=float(gen_cfg.simulation.ssa.sample_interval),
                    burn_in_fraction=float(gen_cfg.simulation.burn_in_fraction),
                    num_replications=_replication_budget(gen_cfg),
                    base_seed=int(gen_cfg.simulation.seed + idx * 1000),
                )
            )

    if ep.include_critical_contract:
        critical_cfg = _contract_cfg(raw_cfg, profile_name, "critical")
        rhos = list(critical_cfg.generalization.rho_boundary_vals)
        if ep.mode != "full":
            rhos = _critical_rhos(rhos, int(ep.quick_critical_max_rhos))
        for idx, rho in enumerate(rhos):
            scenarios.append(
                ScenarioSpec(
                    contract="critical",
                    name=f"critical/rho={rho}",
                    policy_name="calibrated_uas",
                    label=f"critical::Calibrated UAS::{rho:.3f}",
                    alpha=float(critical_cfg.system.alpha),
                    num_servers=critical_cfg.system.num_servers,
                    arrival_rate=float(rho * np.sum(critical_cfg.system.service_rates)),
                    service_rates=list(critical_cfg.system.service_rates),
                    sim_time=float(critical_load_sim_time(critical_cfg, float(rho))),
                    sample_interval=float(critical_cfg.simulation.ssa.sample_interval),
                    burn_in_fraction=float(critical_cfg.simulation.burn_in_fraction),
                    num_replications=_replication_budget(critical_cfg),
                    base_seed=int(critical_cfg.simulation.seed + idx * 1000),
                )
            )

    return scenarios


def _results_from_jax(
    spec: ScenarioSpec,
    cfg: ExperimentConfig,
) -> tuple[list[SimResult], bool]:
    mu = np.asarray(spec.service_rates, dtype=np.float64)
    max_samples = int(spec.sim_time / spec.sample_interval) + 1
    truncated = False
    with warnings.catch_warnings(record=True) as caught:
        times, states, (arrs, deps) = run_replications_jax(
            num_replications=spec.num_replications,
            num_servers=spec.num_servers,
            arrival_rate=spec.arrival_rate,
            service_rates=np.asarray(mu, dtype=np.float32),
            alpha=spec.alpha,
            sim_time=spec.sim_time,
            sample_interval=spec.sample_interval,
            base_seed=spec.base_seed,
            max_samples=max_samples,
            policy_type=policy_name_to_type(spec.policy_name),
            max_events_multiplier=cfg.jax_engine.max_events_safety_multiplier,
            max_events_buffer=cfg.jax_engine.max_events_additive_buffer,
            scan_sampling_chunk=cfg.jax_engine.scan_sampling_chunk,
        )
        truncated = any("truncation" in str(w.message).lower() for w in caught)

    results: list[SimResult] = []
    for rep_idx in range(spec.num_replications):
        np_times = np.asarray(times[rep_idx])
        np_states = np.asarray(states[rep_idx])
        valid_mask = np_times > 0
        valid_mask[0] = True
        valid_len = int(np.sum(valid_mask))
        results.append(
            SimResult(
                times=np_times[:valid_len],
                states=np_states[:valid_len],
                arrival_count=int(arrs[rep_idx]),
                departure_count=int(deps[rep_idx]),
                final_time=float(np_times[valid_len - 1]),
                num_servers=spec.num_servers,
            )
        )
    return results, truncated


def _results_from_numpy(spec: ScenarioSpec) -> tuple[list[SimResult], bool]:
    mu = np.asarray(spec.service_rates, dtype=np.float64)
    policy = build_policy_by_name(spec.policy_name, alpha=spec.alpha, mu=mu)
    max_events = compute_configured_max_events(spec.arrival_rate, mu, spec.sim_time)
    with warnings.catch_warnings(record=True) as caught:
        results = run_replications(
            num_servers=spec.num_servers,
            arrival_rate=spec.arrival_rate,
            service_rates=mu,
            policy=policy,
            sim_time=spec.sim_time,
            sample_interval=spec.sample_interval,
            num_replications=spec.num_replications,
            base_seed=spec.base_seed,
            max_events=max_events,
            progress_desc=f"engine_parity numpy ({spec.name})",
        )
        truncated = any("max_events" in str(w.message).lower() for w in caught)
    return results, truncated


def _summarize_engine(results: list[SimResult], burn_in_fraction: float, truncated: bool) -> EngineMetrics:
    q_totals = []
    final_q_totals = []
    arrival_rates = []
    departure_rates = []
    q_vectors = []

    for result in results:
        avg_q = time_averaged_queue_lengths(result, burn_in_fraction)
        q_vectors.append(avg_q)
        q_totals.append(float(np.sum(avg_q)))
        final_q_totals.append(float(np.sum(result.states[-1])))
        denom = max(float(result.final_time), 1e-8)
        arrival_rates.append(float(result.arrival_count) / denom)
        departure_rates.append(float(result.departure_count) / denom)

    q_arr = np.asarray(q_totals, dtype=np.float64)
    q_vec = np.vstack(q_vectors).mean(axis=0) if q_vectors else np.zeros(1, dtype=np.float64)
    se = float(np.std(q_arr, ddof=1) / np.sqrt(q_arr.size)) if q_arr.size > 1 else 0.0
    return EngineMetrics(
        mean_q_total=float(np.mean(q_arr)),
        std_q_total=float(np.std(q_arr, ddof=1)) if q_arr.size > 1 else 0.0,
        se_q_total=se,
        mean_final_q_total=float(np.mean(final_q_totals)),
        mean_arrival_rate=float(np.mean(arrival_rates)),
        mean_departure_rate=float(np.mean(departure_rates)),
        mean_q_vector=[float(x) for x in q_vec],
        per_rep_q_total=[float(x) for x in q_arr.tolist()],
        truncated=truncated,
    )


def _equivalence_margin(numpy_mean: float, cfg: ExperimentConfig) -> float:
    ep = cfg.engine_parity
    return max(float(ep.equivalence_abs_margin), float(ep.equivalence_rel_margin) * abs(float(numpy_mean)))


def _welch_diff_ci(sample_a: np.ndarray, sample_b: np.ndarray, level: float) -> tuple[float, float, float]:
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    diff = float(np.mean(a) - np.mean(b))
    var_a = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    se = float(np.sqrt(var_a / max(a.size, 1) + var_b / max(b.size, 1)))
    if se == 0.0:
        return diff, diff, diff
    numerator = (var_a / max(a.size, 1) + var_b / max(b.size, 1)) ** 2
    denominator = 0.0
    if a.size > 1 and var_a > 0.0:
        denominator += ((var_a / a.size) ** 2) / (a.size - 1)
    if b.size > 1 and var_b > 0.0:
        denominator += ((var_b / b.size) ** 2) / (b.size - 1)
    df = (numerator / denominator) if denominator > 0 else max(a.size + b.size - 2, 1)
    tcrit = float(stats.t.ppf((1.0 + level) / 2.0, df))
    half_width = tcrit * se
    return diff, diff - half_width, diff + half_width


def _cohen_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return 0.0
    pooled_var = (((a.size - 1) * np.var(a, ddof=1)) + ((b.size - 1) * np.var(b, ddof=1))) / (a.size + b.size - 2)
    if pooled_var <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_var))


def _evaluate_scenario(spec: ScenarioSpec, cfg: ExperimentConfig) -> dict:
    log.info(
        "Engine parity: %s | reps=%d | sim_time=%.1f | alpha=%.2f",
        spec.name,
        spec.num_replications,
        spec.sim_time,
        spec.alpha,
    )
    numpy_results, numpy_truncated = _results_from_numpy(spec)
    jax_results, jax_truncated = _results_from_jax(spec, cfg)

    numpy_metrics = _summarize_engine(numpy_results, spec.burn_in_fraction, numpy_truncated)
    jax_metrics = _summarize_engine(jax_results, spec.burn_in_fraction, jax_truncated)

    numpy_samples = np.asarray(numpy_metrics.per_rep_q_total, dtype=np.float64)
    jax_samples = np.asarray(jax_metrics.per_rep_q_total, dtype=np.float64)
    diff_mean, ci_low, ci_high = _welch_diff_ci(jax_samples, numpy_samples, float(cfg.engine_parity.equivalence_ci_level))
    margin = _equivalence_margin(numpy_metrics.mean_q_total, cfg)
    relative_gap_pct = (diff_mean / max(abs(numpy_metrics.mean_q_total), 1e-8)) * 100.0
    passed = (
        not numpy_metrics.truncated
        and not jax_metrics.truncated
        and ci_low >= -margin
        and ci_high <= margin
    )

    return {
        "contract": spec.contract,
        "scenario": spec.name,
        "policy_name": spec.policy_name,
        "policy_label": spec.label,
        "num_replications": spec.num_replications,
        "sim_time": spec.sim_time,
        "sample_interval": spec.sample_interval,
        "burn_in_fraction": spec.burn_in_fraction,
        "alpha": spec.alpha,
        "arrival_rate": spec.arrival_rate,
        "service_rates": [float(x) for x in spec.service_rates],
        "numpy": asdict(numpy_metrics),
        "jax": asdict(jax_metrics),
        "diff_mean_q_total": float(diff_mean),
        "diff_ci_low": float(ci_low),
        "diff_ci_high": float(ci_high),
        "relative_gap_pct": float(relative_gap_pct),
        "cohen_d": _cohen_d(jax_samples, numpy_samples),
        "equivalence_margin": float(margin),
        "passes_equivalence": bool(passed),
    }


def _write_summary(run_dir: Path, cfg: ExperimentConfig, rows: list[dict]) -> dict:
    pass_count = sum(1 for row in rows if row["passes_equivalence"])
    fail_count = len(rows) - pass_count
    worst_row = max(rows, key=lambda row: abs(row["relative_gap_pct"])) if rows else None
    summary = {
        "mode": cfg.engine_parity.mode,
        "policies": list(cfg.engine_parity.policies),
        "numpy_precision": "float64",
        "jax_precision": str(cfg.jax.precision),
        "num_rows": len(rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "all_passed": fail_count == 0,
        "mean_absolute_relative_gap_pct": float(mean(abs(row["relative_gap_pct"]) for row in rows)) if rows else 0.0,
        "worst_case_scenario": worst_row["scenario"] if worst_row else None,
        "worst_case_policy": worst_row["policy_name"] if worst_row else None,
        "worst_case_relative_gap_pct": float(worst_row["relative_gap_pct"]) if worst_row else 0.0,
    }
    summary_path = metrics_path(run_dir, "engine_parity_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_report(run_dir: Path, summary: dict, rows: list[dict]) -> None:
    lines = [
        "# Engine Parity Report",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- NumPy precision: `{summary['numpy_precision']}`",
        f"- JAX precision: `{summary['jax_precision']}`",
        f"- Rows evaluated: `{summary['num_rows']}`",
        f"- Passed: `{summary['pass_count']}`",
        f"- Failed: `{summary['fail_count']}`",
        f"- All passed: `{summary['all_passed']}`",
        f"- Mean absolute relative gap (%): `{summary['mean_absolute_relative_gap_pct']:.4f}`",
        "",
        "## Rows",
        "",
        "| Contract | Scenario | Policy | NumPy E[Q] | JAX E[Q] | Diff | 90% CI | Margin | Pass |",
        "|----------|----------|--------|------------|----------|------|---------|--------|------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['contract']} | {row['scenario']} | {row['policy_name']} | "
            f"{row['numpy']['mean_q_total']:.4f} | {row['jax']['mean_q_total']:.4f} | "
            f"{row['diff_mean_q_total']:.4f} | [{row['diff_ci_low']:.4f}, {row['diff_ci_high']:.4f}] | "
            f"{row['equivalence_margin']:.4f} | {'PASS' if row['passes_equivalence'] else 'FAIL'} |"
        )
    metadata_path(run_dir, "engine_parity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_scatter(run_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    x = np.array([row["numpy"]["mean_q_total"] for row in rows], dtype=np.float64)
    y = np.array([row["jax"]["mean_q_total"] for row in rows], dtype=np.float64)
    colors = ["#2e7d32" if row["passes_equivalence"] else "#c62828" for row in rows]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, c=colors, s=55, alpha=0.85)
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("NumPy mean E[Q_total]")
    ax.set_ylabel("JAX mean E[Q_total]")
    ax.set_title("Engine Parity: JAX vs NumPy")
    ax.grid(alpha=0.25)
    plot_base = figure_path(run_dir, "engine_parity_scatter")
    fig.tight_layout()
    fig.savefig(plot_base.with_suffix(".png"), dpi=200)
    fig.savefig(plot_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_diffs(run_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda row: abs(row["relative_gap_pct"]), reverse=True)
    labels = [row["scenario"] for row in ordered]
    diffs = np.array([row["diff_mean_q_total"] for row in ordered], dtype=np.float64)
    lows = np.array([row["diff_ci_low"] for row in ordered], dtype=np.float64)
    highs = np.array([row["diff_ci_high"] for row in ordered], dtype=np.float64)
    margins = np.array([row["equivalence_margin"] for row in ordered], dtype=np.float64)
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(ordered) + 1)))
    ax.errorbar(
        diffs,
        y,
        xerr=np.vstack((diffs - lows, highs - diffs)),
        fmt="o",
        color="#1565c0",
        ecolor="#1565c0",
        capsize=3,
    )
    for idx, margin in enumerate(margins):
        ax.axvline(margin, color="#2e7d32", linestyle=":", linewidth=0.8)
        ax.axvline(-margin, color="#2e7d32", linestyle=":", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("JAX - NumPy mean E[Q_total]")
    ax.set_title("Engine Parity: Mean Difference with Equivalence CI")
    ax.grid(alpha=0.25, axis="x")
    plot_base = figure_path(run_dir, "engine_parity_diff_ci")
    fig.tight_layout()
    fig.savefig(plot_base.with_suffix(".png"), dpi=200)
    fig.savefig(plot_base.with_suffix(".pdf"))
    plt.close(fig)


class EngineParityExperiment:
    def __init__(self, cfg: ExperimentConfig, raw_cfg: DictConfig, run_dir: Path, run_logger=None):
        self.cfg = cfg
        self.raw_cfg = raw_cfg
        self.run_dir = run_dir
        self.run_logger = run_logger
        self.profile_name = str(raw_cfg.get("active_profile", "default"))

    def execute(self) -> dict:
        scenarios = _build_scenarios(self.raw_cfg, self.profile_name, self.cfg)
        log.info("Engine parity scenarios: %d", len(scenarios))
        rows: list[dict] = []
        rows_path = metrics_path(self.run_dir, "engine_parity_rows.jsonl")
        for spec in scenarios:
            row = _evaluate_scenario(spec, self.cfg)
            rows.append(row)
            append_metrics_jsonl(row, rows_path)

        summary = _write_summary(self.run_dir, self.cfg, rows)
        _write_report(self.run_dir, summary, rows)
        _plot_scatter(self.run_dir, rows)
        _plot_diffs(self.run_dir, rows)

        log.info(
            "Engine parity complete: %d/%d passed equivalence.",
            summary["pass_count"],
            summary["num_rows"],
        )
        if self.run_logger:
            self.run_logger.log(
                {
                    "engine_parity/pass_count": summary["pass_count"],
                    "engine_parity/fail_count": summary["fail_count"],
                    "engine_parity/mean_absolute_relative_gap_pct": summary["mean_absolute_relative_gap_pct"],
                    "engine_parity/all_passed": float(summary["all_passed"]),
                }
            )
        return summary


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(raw_cfg: DictConfig):
    cfg, resolved_raw_cfg = load_experiment_config(raw_cfg, "engine_parity")
    run_dir, run_id = get_run_config(cfg, "engine_parity", resolved_raw_cfg)
    run_logger = setup_wandb(
        cfg,
        resolved_raw_cfg,
        default_group="engine_parity",
        run_id=run_id,
        run_dir=run_dir,
    )

    log.info("=" * 60)
    log.info("  Engine Parity Verification")
    log.info("=" * 60)

    experiment = EngineParityExperiment(cfg, raw_cfg, run_dir, run_logger)
    return experiment.execute()


if __name__ == "__main__":
    main()
