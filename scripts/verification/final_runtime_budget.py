#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verification.runtime_budgeting import (
    GROUP_A_EXPERIMENTS,
    GROUP_B_EXPERIMENTS,
    GROUP_C_EXPERIMENTS,
    ExperimentTiming,
    FitResult,
    append_jsonl,
    apply_feature_overrides,
    experiment_runtime_features,
    fit_affine_model,
    latest_timing_anchors,
    predict_affine,
    ranking_for_groups,
    read_jsonl,
)


LOG_PATHS = (
    PROJECT_ROOT / "logs" / "debug_logs.md",
    PROJECT_ROOT / "logs" / "small_logs.md",
)


def _anchor_samples(anchors: dict[tuple[str, str], ExperimentTiming], experiment_name: str) -> list[tuple[dict[str, float], float]]:
    samples: list[tuple[dict[str, float], float]] = []
    for profile_name in ("debug", "small"):
        anchor = anchors.get((profile_name, experiment_name))
        if anchor is None:
            continue
        samples.append((experiment_runtime_features(profile_name, experiment_name), anchor.elapsed_seconds))
    return samples


def _calibration_samples(path: Path) -> list[tuple[dict[str, float], float]]:
    rows = []
    for row in read_jsonl(path):
        if row.get("status") != "completed":
            continue
        features = row.get("resolved_features")
        runtime = row.get("wall_time_seconds")
        if isinstance(features, dict) and runtime is not None:
            rows.append((features, float(runtime)))
    return rows


def _fit_reinforce(anchors: dict[tuple[str, str], ExperimentTiming], calibration_dir: Path) -> tuple[FitResult, list[tuple[dict[str, float], float]]]:
    samples = _anchor_samples(anchors, "reinforce_train")
    samples.extend(_calibration_samples(calibration_dir / "reinforce_train.jsonl"))
    fit = fit_affine_model(
        samples,
        feature_names=("train_work", "eval_work"),
        formula="runtime ≈ intercept + c_train * train_work + c_eval * eval_work",
    )
    return fit, samples


def _fit_generalize(anchors: dict[tuple[str, str], ExperimentTiming], calibration_dir: Path) -> tuple[FitResult, list[tuple[dict[str, float], float]]]:
    samples = _anchor_samples(anchors, "generalize")
    samples.extend(_calibration_samples(calibration_dir / "generalize.jsonl"))
    fit = fit_affine_model(
        samples,
        feature_names=("grid_work",),
        formula="runtime ≈ intercept + c_grid * (|scale_vals| * |rho_grid_vals| * replications * sim_time)",
    )
    return fit, samples


def _fit_critical(anchors: dict[tuple[str, str], ExperimentTiming], calibration_dir: Path) -> tuple[FitResult, list[tuple[dict[str, float], float]]]:
    samples = _anchor_samples(anchors, "critical")
    samples.extend(_calibration_samples(calibration_dir / "critical.jsonl"))
    fit = fit_affine_model(
        samples,
        feature_names=("critical_work",),
        formula="runtime ≈ intercept + c_crit * (replications * Σ critical_load_sim_time(rho))",
    )
    return fit, samples


def _fit_stress(anchors: dict[tuple[str, str], ExperimentTiming], calibration_dir: Path) -> tuple[FitResult, list[tuple[dict[str, float], float]]]:
    samples = _anchor_samples(anchors, "stress")
    samples.extend(_calibration_samples(calibration_dir / "stress.jsonl"))
    fit = fit_affine_model(
        samples,
        feature_names=("stage_work",),
        formula="runtime ≈ intercept + c_stress * stage_work",
    )
    return fit, samples


def _ablation_ratio(anchors: dict[tuple[str, str], ExperimentTiming]) -> tuple[float, str]:
    ratios: list[float] = []
    for profile_name in ("debug", "small"):
        abl = anchors.get((profile_name, "ablation"))
        reinf = anchors.get((profile_name, "reinforce_train"))
        if abl is None or reinf is None or reinf.elapsed_seconds <= 0:
            continue
        ratios.append(float(abl.elapsed_seconds / reinf.elapsed_seconds))
    if not ratios:
        return 3.2, "low"
    if len(ratios) >= 2:
        return float(sum(ratios) / len(ratios)), "medium"
    return ratios[0], "low"


def _estimate_ablation(
    anchors: dict[tuple[str, str], ExperimentTiming],
    reinforce_fit: FitResult,
) -> dict[str, Any]:
    ratio, confidence = _ablation_ratio(anchors)
    final_features = experiment_runtime_features("final_experiment", "ablation")
    reinforce_features = experiment_runtime_features("final_experiment", "reinforce_train")
    reinforce_estimate = predict_affine(reinforce_features, reinforce_fit)
    predicted = ratio * reinforce_estimate
    return {
        "formula": f"runtime ≈ {ratio:.3f} * reinforce_train_estimate",
        "confidence": confidence,
        "ratio": ratio,
        "features": final_features,
        "predicted_seconds": predicted,
    }


def _candidate_records(
    experiment_name: str,
    fit: FitResult,
    budget_seconds: float,
) -> list[dict[str, Any]]:
    current = dict(experiment_runtime_features("final_experiment", experiment_name))
    candidates: list[dict[str, Any]] = []

    def maybe_add(candidate: dict[str, float]) -> None:
        predicted = predict_affine(candidate, fit)
        if predicted <= budget_seconds:
            candidates.append(
                {
                    "experiment": experiment_name,
                    "predicted_seconds": predicted,
                    "predicted_minutes": predicted / 60.0,
                    "feature_values": candidate,
                    "hydra_overrides": apply_feature_overrides(experiment_name, candidate),
                }
            )

    if experiment_name == "reinforce_train":
        eval_batches_vals = [int(current["eval_batches"]), max(1, int(current["eval_batches"] // 2)), 1]
        eval_traj_vals = [int(current["eval_trajs_per_batch"]), max(1, int(current["eval_trajs_per_batch"] // 2)), 1]
        epoch_vals = [int(current["train_epochs"]), max(1, int(round(current["train_epochs"] * 0.75))), max(1, int(round(current["train_epochs"] * 0.5)))]
        batch_vals = [int(current["batch_size"]), max(1, int(current["batch_size"] // 2))]
        sim_vals = [float(current["sim_time"]), max(500.0, current["sim_time"] * 0.75), max(500.0, current["sim_time"] * 0.5)]
        seen: set[tuple[float, ...]] = set()
        for eb in eval_batches_vals:
            for et in eval_traj_vals:
                for ep in epoch_vals:
                    for bs in batch_vals:
                        for st in sim_vals:
                            candidate = dict(current)
                            candidate.update(
                                {
                                    "eval_batches": float(eb),
                                    "eval_trajs_per_batch": float(et),
                                    "train_epochs": float(ep),
                                    "batch_size": float(bs),
                                    "sim_time": float(st),
                                    "train_work": float(ep) * float(bs) * float(st),
                                    "eval_work": float(eb) * float(et) * float(st),
                                }
                            )
                            key = (candidate["eval_batches"], candidate["eval_trajs_per_batch"], candidate["train_epochs"], candidate["batch_size"], candidate["sim_time"])
                            if key in seen:
                                continue
                            seen.add(key)
                            maybe_add(candidate)

    elif experiment_name == "generalize":
        rep_vals = [int(current["replications"]), max(1, int(current["replications"] // 2)), max(1, int(current["replications"] // 4))]
        scale_vals = [int(current["scale_count"]), min(3, int(current["scale_count"])), min(2, int(current["scale_count"]))]
        rho_vals = [int(current["rho_grid_count"]), min(3, int(current["rho_grid_count"])), min(2, int(current["rho_grid_count"]))]
        sim_vals = [float(current["sim_time"]), max(500.0, current["sim_time"] * 0.75), max(500.0, current["sim_time"] * 0.5)]
        seen: set[tuple[float, ...]] = set()
        for rep in rep_vals:
            for scale_count in scale_vals:
                for rho_count in rho_vals:
                    for sim_time in sim_vals:
                        candidate = dict(current)
                        cells = float(scale_count * rho_count)
                        candidate.update(
                            {
                                "replications": float(rep),
                                "scale_count": float(scale_count),
                                "rho_grid_count": float(rho_count),
                                "sim_time": float(sim_time),
                                "cells": cells,
                                "grid_work": cells * float(rep) * float(sim_time),
                            }
                        )
                        key = (candidate["replications"], candidate["scale_count"], candidate["rho_grid_count"], candidate["sim_time"])
                        if key in seen:
                            continue
                        seen.add(key)
                        maybe_add(candidate)

    elif experiment_name == "critical":
        rep_vals = [int(current["replications"]), max(1, int(current["replications"] // 2)), max(1, int(current["replications"] // 4))]
        rho_vals = [int(current["rho_count"]), min(3, int(current["rho_count"])), min(2, int(current["rho_count"]))]
        horizon_per_rho = float(current["critical_horizon_sum"] / max(current["rho_count"], 1.0))
        seen: set[tuple[float, ...]] = set()
        for rep in rep_vals:
            for rho_count in rho_vals:
                horizon_sum = float(rho_count) * horizon_per_rho
                candidate = dict(current)
                candidate.update(
                    {
                        "replications": float(rep),
                        "rho_count": float(rho_count),
                        "critical_horizon_sum": horizon_sum,
                        "critical_work": float(rep) * horizon_sum,
                    }
                )
                key = (candidate["replications"], candidate["rho_count"])
                if key in seen:
                    continue
                seen.add(key)
                maybe_add(candidate)

    elif experiment_name == "stress":
        rep_vals = [int(current["replications"]), max(1, int(current["replications"] // 2))]
        for rep in rep_vals:
            candidate = dict(current)
            base_stage = current["stage_work"] / max(current["replications"], 1.0)
            candidate.update(
                {
                    "replications": float(rep),
                    "stage_work": float(rep) * base_stage,
                }
            )
            maybe_add(candidate)

    candidates.sort(key=lambda row: row["predicted_seconds"])
    return candidates[:5]


def _current_estimate_record(experiment_name: str, fit: FitResult) -> dict[str, Any]:
    features = experiment_runtime_features("final_experiment", experiment_name)
    predicted = predict_affine(features, fit)
    return {
        "experiment": experiment_name,
        "formula": fit.formula,
        "coefficients": fit.coefficients,
        "residual_rmse": fit.residual_rmse,
        "confidence": fit.confidence,
        "calibration_count": fit.calibration_count,
        "features": features,
        "predicted_seconds": predicted,
        "predicted_minutes": predicted / 60.0,
    }


def build_runtime_plan(calibration_dir: Path, standalone_budget_minutes: float) -> tuple[dict[str, Any], dict[str, Any], str]:
    anchors = latest_timing_anchors([path for path in LOG_PATHS if path.exists()])

    reinforce_fit, reinforce_samples = _fit_reinforce(anchors, calibration_dir)
    generalize_fit, generalize_samples = _fit_generalize(anchors, calibration_dir)
    critical_fit, critical_samples = _fit_critical(anchors, calibration_dir)
    stress_fit, stress_samples = _fit_stress(anchors, calibration_dir)
    ablation_estimate = _estimate_ablation(anchors, reinforce_fit)

    current = {
        "group_a": list(GROUP_A_EXPERIMENTS),
        "group_b": list(GROUP_B_EXPERIMENTS),
        "group_c": list(GROUP_C_EXPERIMENTS),
        "log_anchors": {
            f"{config}:{experiment}": {
                "elapsed_seconds": anchor.elapsed_seconds,
                "source_path": anchor.source_path,
            }
            for (config, experiment), anchor in anchors.items()
        },
        "experiments": {
            "reinforce_train": _current_estimate_record("reinforce_train", reinforce_fit),
            "generalize": _current_estimate_record("generalize", generalize_fit),
            "critical": _current_estimate_record("critical", critical_fit),
            "stress": _current_estimate_record("stress", stress_fit),
            "ablation": ablation_estimate,
        },
    }

    budget_seconds = standalone_budget_minutes * 60.0
    candidates = {
        "reinforce_train": _candidate_records("reinforce_train", reinforce_fit, budget_seconds),
        "generalize": _candidate_records("generalize", generalize_fit, budget_seconds),
        "critical": _candidate_records("critical", critical_fit, budget_seconds),
        "stress": _candidate_records("stress", stress_fit, 45.0 * 60.0),
    }

    if candidates["stress"]:
        current["group_ranking"] = ranking_for_groups(
            {"stress": current["experiments"]["stress"]["predicted_seconds"]},
            standalone_threshold_seconds=45.0 * 60.0,
        )

    lines = [
        "# Final Runtime Summary",
        "",
        f"- Standalone budget target: {standalone_budget_minutes:.1f} minutes",
        f"- Calibration directory: `{calibration_dir}`",
        "",
        "## Current Final Estimates",
    ]
    for name in ("reinforce_train", "generalize", "critical", "ablation", "stress"):
        record = current["experiments"][name]
        lines.append(
            f"- `{name}`: {record['predicted_minutes']:.2f} min"
            if "predicted_minutes" in record
            else f"- `{name}`: {record['predicted_seconds'] / 60.0:.2f} min"
        )
        lines.append(f"  formula: `{record['formula']}`")
        lines.append(f"  confidence: `{record['confidence']}`")
    lines.append("")
    lines.append("## Budget Candidates")
    for name, rows in candidates.items():
        if not rows:
            lines.append(f"- `{name}`: no candidate under budget from current search space")
            continue
        best = rows[0]
        lines.append(
            f"- `{name}` best candidate: {best['predicted_minutes']:.2f} min using "
            f"`{' '.join(best['hydra_overrides'])}`"
        )
    return current, candidates, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid runtime estimator and budget solver for final_experiment")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "runtime_calibration",
        help="Directory containing calibration JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "final_runtime_budget",
        help="Directory for runtime-estimator artifacts",
    )
    parser.add_argument(
        "--standalone-budget-minutes",
        type=float,
        default=240.0,
        help="Per-experiment standalone Colab budget in minutes",
    )
    args = parser.parse_args()

    current, candidates, summary = build_runtime_plan(
        calibration_dir=args.calibration_dir,
        standalone_budget_minutes=args.standalone_budget_minutes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "current_final_estimate.json").write_text(json.dumps(current, indent=2), encoding="utf-8")
    (args.output_dir / "budget_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    (args.output_dir / "runtime_summary.md").write_text(summary, encoding="utf-8")

    print("=" * 58)
    print(" Final Runtime Budget Planner")
    print("=" * 58)
    print(f" Wrote: {args.output_dir / 'current_final_estimate.json'}")
    print(f" Wrote: {args.output_dir / 'budget_candidates.json'}")
    print(f" Wrote: {args.output_dir / 'runtime_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
