from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from omegaconf import OmegaConf

from gibbsq.core.config import (
    critical_load_sim_time,
    load_experiment_config,
    load_experiment_config_chain,
    resolve_experiment_config_chain,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"

GROUP_A_EXPERIMENTS = (
    "check_configs",
    "reinforce_check",
    "drift",
    "sweep",
    "stress",
    "bc_train",
)
GROUP_B_EXPERIMENTS = (
    "reinforce_train",
    "generalize",
    "critical",
    "ablation",
)
GROUP_C_EXPERIMENTS = (
    "policy",
    "stats",
)
ALL_GROUPED_EXPERIMENTS = GROUP_A_EXPERIMENTS + GROUP_B_EXPERIMENTS + GROUP_C_EXPERIMENTS

PIPELINE_COMMAND_RE = re.compile(
    r"reproduction_pipeline\.py\s+--config-name\s+(?P<config>[A-Za-z0-9_]+)"
)
START_EXPERIMENT_RE = re.compile(r"^ Starting Experiment: (?P<experiment>[a-z_]+)\s*$")
CONFIG_PROFILE_RE = re.compile(r"^ Config Profile: (?P<config>[A-Za-z0-9_]+)\s*$")
FINISH_EXPERIMENT_RE = re.compile(r"^\[Experiment '(?P<experiment>[a-z_]+)' Finished\]\s*$")
ELAPSED_RE = re.compile(r"^\s*Elapsed Duration: (?P<seconds>[0-9.]+)s\s*$")
PIPELINE_STATUS_RE = re.compile(r"^\s*Pipeline Status: (?P<status>.+?)\s*$")


@dataclass(frozen=True)
class ExperimentTiming:
    config_name: str
    experiment: str
    elapsed_seconds: float
    source_path: str


@dataclass(frozen=True)
class PipelineSection:
    config_name: str
    source_path: str
    timings: dict[str, float]
    completed: bool
    ordinal: int


@dataclass(frozen=True)
class FitResult:
    formula: str
    coefficients: dict[str, float]
    residual_rmse: float
    confidence: str
    calibration_count: int


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_profile_raw(profile_name: str) -> Any:
    raw = OmegaConf.load(CONFIGS_DIR / f"{profile_name}.yaml")
    OmegaConf.update(raw, "active_profile", profile_name, force_add=True)
    return raw


def resolve_experiment_cfg(profile_name: str, experiment_name: str):
    raw = _load_profile_raw(profile_name)
    if experiment_name == "ablation":
        return load_experiment_config_chain(raw, ["reinforce_train", "ablation"], profile_name=profile_name)
    return load_experiment_config(raw, experiment_name, profile_name=profile_name)


def resolve_raw_experiment_chain(profile_name: str, experiment_names: Sequence[str]):
    raw = _load_profile_raw(profile_name)
    return resolve_experiment_config_chain(raw, experiment_names, profile_name=profile_name)


def parse_pipeline_sections(paths: Sequence[Path]) -> list[PipelineSection]:
    sections: list[PipelineSection] = []
    ordinal = 0
    for path in paths:
        current_config: str | None = None
        current_timings: dict[str, float] = {}
        pending_experiment: str | None = None
        current_completed = False
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            match = PIPELINE_COMMAND_RE.search(line)
            if match:
                if current_config is not None:
                    sections.append(
                        PipelineSection(
                            config_name=current_config,
                            source_path=str(path),
                            timings=dict(current_timings),
                            completed=current_completed,
                            ordinal=ordinal,
                        )
                    )
                    ordinal += 1
                current_config = match.group("config")
                current_timings = {}
                pending_experiment = None
                current_completed = False
                continue

            match = FINISH_EXPERIMENT_RE.match(line)
            if match:
                pending_experiment = match.group("experiment")
                continue

            match = ELAPSED_RE.match(line)
            if match and pending_experiment and current_config is not None:
                current_timings[pending_experiment] = float(match.group("seconds"))
                pending_experiment = None
                continue

            match = PIPELINE_STATUS_RE.match(line)
            if match and current_config is not None:
                status = match.group("status").strip().lower()
                current_completed = status == "completed"

        if current_config is not None:
            sections.append(
                PipelineSection(
                    config_name=current_config,
                    source_path=str(path),
                    timings=dict(current_timings),
                    completed=current_completed,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    return sections


def latest_complete_sections_by_config(paths: Sequence[Path]) -> dict[str, PipelineSection]:
    sections = [section for section in parse_pipeline_sections(paths) if section.completed]
    latest: dict[str, PipelineSection] = {}
    for section in sections:
        latest[section.config_name] = section
    return latest


def latest_timing_anchors(paths: Sequence[Path]) -> dict[tuple[str, str], ExperimentTiming]:
    sections = latest_complete_sections_by_config(paths)
    anchors: dict[tuple[str, str], ExperimentTiming] = {}
    for config_name, section in sections.items():
        for experiment, elapsed in section.timings.items():
            anchors[(config_name, experiment)] = ExperimentTiming(
                config_name=config_name,
                experiment=experiment,
                elapsed_seconds=elapsed,
                source_path=section.source_path,
            )
    return anchors


def stage_profile_path(output_root: Path, profile_name: str, experiment_name: str) -> Path | None:
    experiment_root = output_root / profile_name / experiment_name
    if not experiment_root.exists():
        return None
    pattern = {
        "reinforce_train": "reinforce_stage_profile.json",
    }.get(experiment_name)
    if pattern is None:
        return None
    candidates = sorted(experiment_root.glob(f"run_*/{pattern}"))
    return candidates[-1] if candidates else None


def load_stage_profile(output_root: Path, profile_name: str, experiment_name: str) -> dict[str, Any] | None:
    profile_path = stage_profile_path(output_root, profile_name, experiment_name)
    if profile_path is None:
        return None
    return json.loads(profile_path.read_text(encoding="utf-8"))


def experiment_runtime_features(profile_name: str, experiment_name: str) -> dict[str, float]:
    cfg, _ = resolve_experiment_cfg(profile_name, experiment_name)
    if experiment_name == "reinforce_train":
        eval_work = (
            float(cfg.neural_training.eval_batches)
            * float(cfg.neural_training.eval_trajs_per_batch)
            * float(cfg.simulation.ssa.sim_time)
        )
        train_work = (
            float(cfg.train_epochs)
            * float(cfg.batch_size)
            * float(cfg.simulation.ssa.sim_time)
        )
        return {
            "setup_unit": 1.0,
            "train_work": train_work,
            "eval_work": eval_work,
            "train_epochs": float(cfg.train_epochs),
            "batch_size": float(cfg.batch_size),
            "sim_time": float(cfg.simulation.ssa.sim_time),
            "eval_batches": float(cfg.neural_training.eval_batches),
            "eval_trajs_per_batch": float(cfg.neural_training.eval_trajs_per_batch),
            "checkpoint_freq": float(cfg.neural_training.checkpoint_freq),
        }

    if experiment_name == "generalize":
        scale_count = float(len(cfg.generalization.scale_vals))
        rho_grid_count = float(len(cfg.generalization.rho_grid_vals))
        cells = float(scale_count * rho_grid_count)
        replications = float(cfg.simulation.num_replications)
        sim_time = float(cfg.simulation.ssa.sim_time)
        return {
            "setup_unit": 1.0,
            "scale_count": scale_count,
            "rho_grid_count": rho_grid_count,
            "cells": cells,
            "replications": replications,
            "sim_time": sim_time,
            "grid_work": cells * replications * sim_time,
        }

    if experiment_name == "critical":
        horizons = [float(critical_load_sim_time(cfg, float(rho))) for rho in cfg.generalization.rho_boundary_vals]
        horizon_sum = float(sum(horizons))
        return {
            "setup_unit": 1.0,
            "replications": float(cfg.simulation.num_replications),
            "rho_count": float(len(cfg.generalization.rho_boundary_vals)),
            "critical_horizon_sum": horizon_sum,
            "critical_work": float(cfg.simulation.num_replications) * horizon_sum,
        }

    if experiment_name == "stress":
        critical_horizons = [
            float(critical_load_sim_time(cfg, float(rho)))
            for rho in cfg.stress.critical_rhos
        ]
        stage_work = (
            len(cfg.stress.n_values) * float(cfg.stress.massive_n_sim_time)
            + sum(critical_horizons)
            + float(cfg.stress.heterogeneity_sim_time)
        )
        return {
            "setup_unit": 1.0,
            "replications": float(cfg.simulation.num_replications),
            "n_values": float(len(cfg.stress.n_values)),
            "critical_rhos": float(len(cfg.stress.critical_rhos)),
            "massive_n_sim_time": float(cfg.stress.massive_n_sim_time),
            "heterogeneity_sim_time": float(cfg.stress.heterogeneity_sim_time),
            "stage_work": float(cfg.simulation.num_replications) * stage_work,
        }

    if experiment_name == "ablation":
        base_features = experiment_runtime_features(profile_name, "reinforce_train")
        evaluation_work = 4.0 * float(cfg.simulation.num_replications) * float(cfg.simulation.ssa.sim_time)
        return {
            "setup_unit": 1.0,
            "embedded_reinforce_work": 3.0 * base_features["train_work"],
            "embedded_eval_work": 3.0 * base_features["eval_work"],
            "variant_eval_work": evaluation_work,
            "replications": float(cfg.simulation.num_replications),
            "sim_time": float(cfg.simulation.ssa.sim_time),
        }

    if experiment_name == "policy":
        return {
            "setup_unit": 1.0,
            "replications": float(cfg.simulation.num_replications),
            "sim_time": float(cfg.simulation.ssa.sim_time),
            "policy_work": 4.0 * float(cfg.simulation.num_replications) * float(cfg.simulation.ssa.sim_time),
        }

    if experiment_name == "stats":
        return {
            "setup_unit": 1.0,
            "replications": float(cfg.simulation.num_replications),
            "sim_time": float(cfg.simulation.ssa.sim_time),
            "stats_work": 2.0 * float(cfg.simulation.num_replications) * float(cfg.simulation.ssa.sim_time),
        }

    cfg, _ = resolve_experiment_cfg(profile_name, experiment_name)
    return {
        "setup_unit": 1.0,
        "replications": float(cfg.simulation.num_replications),
        "sim_time": float(cfg.simulation.ssa.sim_time),
    }


def fit_affine_model(
    samples: Sequence[tuple[dict[str, float], float]],
    feature_names: Sequence[str],
    formula: str,
) -> FitResult:
    if not samples:
        return FitResult(formula=formula, coefficients={}, residual_rmse=float("nan"), confidence="low", calibration_count=0)

    x_rows = []
    y = []
    for features, runtime in samples:
        x_rows.append([1.0] + [float(features.get(name, 0.0)) for name in feature_names])
        y.append(float(runtime))
    x = np.array(x_rows, dtype=float)
    y_arr = np.array(y, dtype=float)

    if len(samples) == 1:
        denom = max(abs(x[0, 1]) if x.shape[1] > 1 else 1.0, 1e-9)
        slope = y_arr[0] / denom if x.shape[1] > 1 else y_arr[0]
        coeffs = {"intercept": 0.0}
        if x.shape[1] > 1:
            coeffs[feature_names[0]] = float(slope)
            for name in feature_names[1:]:
                coeffs[name] = 0.0
        residual = 0.0
        confidence = "low"
        return FitResult(formula=formula, coefficients=coeffs, residual_rmse=residual, confidence=confidence, calibration_count=1)

    beta, *_ = np.linalg.lstsq(x, y_arr, rcond=None)
    preds = x @ beta
    residual = float(np.sqrt(np.mean((preds - y_arr) ** 2)))
    if len(samples) >= 5 and residual < max(5.0, 0.15 * float(np.mean(y_arr))):
        confidence = "high"
    elif residual < max(15.0, 0.30 * float(np.mean(y_arr))):
        confidence = "medium"
    else:
        confidence = "low"
    coeffs = {"intercept": float(beta[0])}
    for idx, name in enumerate(feature_names, start=1):
        coeffs[name] = float(beta[idx])
    return FitResult(
        formula=formula,
        coefficients=coeffs,
        residual_rmse=residual,
        confidence=confidence,
        calibration_count=len(samples),
    )


def predict_affine(features: dict[str, float], fit: FitResult) -> float:
    total = fit.coefficients.get("intercept", 0.0)
    for name, coefficient in fit.coefficients.items():
        if name == "intercept":
            continue
        total += coefficient * float(features.get(name, 0.0))
    return max(0.0, float(total))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _coerce_override_value(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def apply_feature_overrides(
    experiment_name: str,
    features: dict[str, float],
) -> list[str]:
    overrides: list[str] = []
    if experiment_name == "reinforce_train":
        mappings = {
            "train_epochs": "train_epochs",
            "batch_size": "batch_size",
            "sim_time": "simulation.ssa.sim_time",
            "eval_batches": "neural_training.eval_batches",
            "eval_trajs_per_batch": "neural_training.eval_trajs_per_batch",
            "checkpoint_freq": "neural_training.checkpoint_freq",
        }
        for name, key in mappings.items():
            if name in features:
                overrides.append(f"{key}={_coerce_override_value(float(features[name]))}")
        return overrides

    if experiment_name == "generalize":
        if "replications" in features:
            overrides.append(f"simulation.num_replications={_coerce_override_value(features['replications'])}")
        if "sim_time" in features:
            overrides.append(f"simulation.ssa.sim_time={_coerce_override_value(features['sim_time'])}")
        if "scale_count" in features:
            raw = _load_profile_raw("final_experiment")
            cfg, _ = load_experiment_config(raw, "generalize", profile_name="final_experiment")
            subset = cfg.generalization.scale_vals[: int(features["scale_count"])]
            overrides.append(f"generalization.scale_vals={json.dumps(subset)}")
        if "rho_grid_count" in features:
            raw = _load_profile_raw("final_experiment")
            cfg, _ = load_experiment_config(raw, "generalize", profile_name="final_experiment")
            subset = cfg.generalization.rho_grid_vals[: int(features["rho_grid_count"])]
            overrides.append(f"generalization.rho_grid_vals={json.dumps(subset)}")
        return overrides

    if experiment_name == "critical":
        if "replications" in features:
            overrides.append(f"simulation.num_replications={_coerce_override_value(features['replications'])}")
        if "rho_count" in features:
            raw = _load_profile_raw("final_experiment")
            cfg, _ = load_experiment_config(raw, "critical", profile_name="final_experiment")
            subset = cfg.generalization.rho_boundary_vals[: int(features["rho_count"])]
            overrides.append(f"generalization.rho_boundary_vals={json.dumps(subset)}")
        return overrides

    if experiment_name == "stress":
        if "replications" in features:
            overrides.append(f"simulation.num_replications={_coerce_override_value(features['replications'])}")
        return overrides

    if experiment_name == "ablation":
        if "replications" in features:
            overrides.append(f"simulation.num_replications={_coerce_override_value(features['replications'])}")
        return overrides

    return overrides


def ranking_for_groups(
    timings: dict[str, float],
    standalone_threshold_seconds: float = 45.0 * 60.0,
) -> dict[str, str]:
    groups = {alias: "A" for alias in GROUP_A_EXPERIMENTS}
    groups.update({alias: "B" for alias in GROUP_B_EXPERIMENTS})
    groups.update({alias: "C" for alias in GROUP_C_EXPERIMENTS})
    if timings.get("stress", 0.0) > standalone_threshold_seconds:
        groups["stress"] = "B"
    return groups


def _three_point_ints(base: int, minimum: int = 1) -> list[int]:
    values = {
        max(minimum, int(math.floor(base * 0.5))),
        max(minimum, int(base)),
        max(minimum, int(math.ceil(base * 1.5))),
    }
    return sorted(values)


def _three_point_floats(base: float, minimum: float) -> list[float]:
    values = {
        max(minimum, float(base * 0.5)),
        max(minimum, float(base)),
        max(minimum, float(base * 1.5)),
    }
    return sorted(values)


def default_calibration_matrix(profile_name: str, experiment_name: str) -> list[dict[str, float]]:
    cfg, _ = resolve_experiment_cfg(profile_name, experiment_name)
    rows: list[dict[str, float]] = []

    if experiment_name == "reinforce_train":
        rows.extend(
            [
                {"probe_case": 1.0},
                {"batch_size": float(max(2, int(cfg.batch_size // 2))), "probe_case": 2.0},
                {"batch_size": float(cfg.batch_size), "probe_case": 3.0},
                {"sim_time": float(max(500.0, cfg.simulation.ssa.sim_time * 0.5)), "probe_case": 4.0},
                {"sim_time": float(cfg.simulation.ssa.sim_time), "probe_case": 5.0},
                {
                    "eval_batches": 1.0,
                    "eval_trajs_per_batch": 1.0,
                    "probe_case": 6.0,
                },
            ]
        )
        return rows

    if experiment_name == "generalize":
        scale_counts = sorted({min(len(cfg.generalization.scale_vals), v) for v in (2, 3)})
        rho_counts = sorted({min(len(cfg.generalization.rho_grid_vals), v) for v in (2, 3)})
        rows.extend(
            [
                {"replications": 2.0, "scale_count": float(scale_counts[0]), "rho_grid_count": float(rho_counts[0]), "probe_case": 1.0},
                {"replications": 4.0, "scale_count": float(scale_counts[0]), "rho_grid_count": float(rho_counts[0]), "probe_case": 2.0},
                {
                    "replications": 2.0,
                    "scale_count": float(scale_counts[-1]),
                    "rho_grid_count": float(rho_counts[0]),
                    "probe_case": 3.0,
                },
                {
                    "replications": 2.0,
                    "scale_count": float(scale_counts[0]),
                    "rho_grid_count": float(rho_counts[-1]),
                    "probe_case": 4.0,
                },
                {
                    "replications": 2.0,
                    "scale_count": float(scale_counts[0]),
                    "rho_grid_count": float(rho_counts[0]),
                    "sim_time": float(max(1000.0, cfg.simulation.ssa.sim_time * 0.5)),
                    "probe_case": 5.0,
                },
            ]
        )
        return rows

    if experiment_name == "critical":
        rho_counts = sorted({min(len(cfg.generalization.rho_boundary_vals), v) for v in (2, 3, len(cfg.generalization.rho_boundary_vals))})
        rows.extend(
            [
                {"replications": 2.0, "rho_count": float(rho_counts[0]), "probe_case": 1.0},
                {"replications": 4.0, "rho_count": float(rho_counts[0]), "probe_case": 2.0},
                {"replications": 2.0, "rho_count": float(rho_counts[-1]), "probe_case": 3.0},
            ]
        )
        return rows

    if experiment_name == "stress":
        rows.extend(
            [
                {"replications": 2.0, "probe_case": 1.0},
                {"replications": 4.0, "probe_case": 2.0},
            ]
        )
        return rows

    if experiment_name == "ablation":
        rows.append({"replications": min(float(cfg.simulation.num_replications), 2.0), "probe_case": 1.0})
        return rows

    return rows


def default_probe_overrides(profile_name: str, experiment_name: str) -> list[str]:
    cfg, _ = resolve_experiment_cfg(profile_name, experiment_name)
    if experiment_name == "reinforce_train":
        return [
            "train_epochs=1",
            f"batch_size={max(2, min(int(cfg.batch_size), 8))}",
            f"simulation.ssa.sim_time={_coerce_override_value(max(500.0, min(float(cfg.simulation.ssa.sim_time), 1500.0)))}",
            "neural_training.eval_batches=1",
            "neural_training.eval_trajs_per_batch=1",
            f"neural_training.checkpoint_freq={max(1, int(cfg.neural_training.checkpoint_freq))}",
        ]
    if experiment_name == "generalize":
        scale_vals = cfg.generalization.scale_vals[:2]
        rho_vals = cfg.generalization.rho_grid_vals[:2]
        return [
            "simulation.num_replications=2",
            f"simulation.ssa.sim_time={_coerce_override_value(max(1000.0, min(float(cfg.simulation.ssa.sim_time), 10000.0)))}",
            f"generalization.scale_vals={json.dumps(scale_vals)}",
            f"generalization.rho_grid_vals={json.dumps(rho_vals)}",
        ]
    if experiment_name == "critical":
        rho_vals = cfg.generalization.rho_boundary_vals[:2]
        return [
            "simulation.num_replications=2",
            f"generalization.rho_boundary_vals={json.dumps(rho_vals)}",
        ]
    if experiment_name == "stress":
        n_values = cfg.stress.n_values[:2]
        critical_rhos = cfg.stress.critical_rhos[:2]
        return [
            "simulation.num_replications=2",
            f"stress.n_values={json.dumps(n_values)}",
            f"stress.critical_rhos={json.dumps(critical_rhos)}",
        ]
    if experiment_name == "ablation":
        return [
            "simulation.num_replications=2",
        ]
    return []
