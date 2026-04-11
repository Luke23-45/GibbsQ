from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import matplotlib
import numpy as np
from omegaconf import DictConfig, OmegaConf

from gibbsq.core.config import load_experiment_config
from gibbsq.core.pretraining import DEFAULT_BC_DATA_CONFIG, extract_bc_data_config
from gibbsq.utils.logging import get_run_config
from gibbsq.utils.run_artifacts import metrics_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARITY_ORDER = {"FAILED": 0, "BRONZE": 1, "SILVER": 2, "GOLD": 3}
MIN_PARITY_BY_NAME = {"FAILED": 0, "BRONZE": 1, "SILVER": 2, "GOLD": 3}

ROUND_SEQUENCE = (
    "bc",
    "reinforce",
    "domain_randomization",
    "architecture",
)

FROZEN_CONTRACT_PATHS = [
    "system.*",
    "policy benchmark definitions",
    "verification.*",
    "evaluation metric semantics",
]

SEARCHABLE_KNOBS = {
    "bc": [
        "neural_training.bc_num_steps",
        "neural_training.bc_lr",
        "neural_training.bc_label_smoothing",
        "bc_data.rhos",
        "bc_data.mu_scales",
        "bc_data.expert_sim_time",
        "bc_data.augmentation_noise_min",
        "bc_data.augmentation_noise_max",
    ],
    "reinforce": [
        "neural.actor_lr",
        "neural.critic_lr",
        "neural.weight_decay",
        "neural.clip_global_norm",
        "neural.lr_decay_rate",
        "neural.entropy_bonus",
        "neural.entropy_final",
        "neural_training.shake_scale",
        "neural_training.gae_lambda",
        "batch_size",
        "train_epochs",
        "simulation.ssa.sim_time",
    ],
    "domain_randomization": [
        "domain_randomization.enabled",
        "domain_randomization.rho_min",
        "domain_randomization.rho_max",
        "domain_randomization.phases",
    ],
    "architecture": [
        "neural.hidden_size",
        "neural.preprocessing",
        "neural.init_type",
        "neural.rho_input_scale",
        "neural.use_rho",
        "neural.use_service_rates",
    ],
}

BUDGET_KNOBS = [
    "simulation.num_replications",
    "simulation.ssa.sim_time",
    "neural_training.eval_batches",
    "neural_training.eval_trajs_per_batch",
    "neural_training.checkpoint_freq",
]

STAGE_PRESETS = {
    "A": {
        "profile_name": "small",
        "seed_count": 3,
        "candidate_count": 12,
        "promote_top_k": 3,
        "suite": ["policy", "generalize"],
    },
    "B": {
        "profile_name": "default",
        "seed_count": 5,
        "candidate_count": 8,
        "promote_top_k": 3,
        "suite": ["policy", "generalize", "critical"],
    },
    "C": {
        "profile_name": "final_experiment",
        "seed_count": 8,
        "candidate_count": 4,
        "promote_top_k": 2,
        "suite": ["policy", "generalize", "critical", "stress"],
    },
}

BC_RHO_LIBRARY = [
    [0.35, 0.55, 0.75],
    [0.45, 0.65, 0.85],
    [0.55, 0.75, 0.9],
]
BC_MU_SCALE_LIBRARY = [
    [0.5, 1.0, 2.0],
    [0.75, 1.0, 1.5],
    [0.5, 1.0, 1.5, 2.0],
]
DR_LIBRARY = [
    {
        "label": "dr_off",
        "overrides": {
            "domain_randomization.enabled": False,
            "domain_randomization.phases": [],
        },
    },
    {
        "label": "dr_conservative",
        "overrides": {
            "domain_randomization.enabled": True,
            "domain_randomization.rho_min": 0.40,
            "domain_randomization.rho_max": 0.85,
            "domain_randomization.phases": [
                {"rho_min": 0.45, "rho_max": 0.70, "epochs": 10, "horizon": 300},
                {"rho_min": 0.50, "rho_max": 0.82, "epochs": 20, "horizon": 1200},
                {"rho_min": 0.60, "rho_max": 0.85, "epochs": 20, "horizon": 3000},
            ],
        },
    },
    {
        "label": "dr_balanced",
        "overrides": {
            "domain_randomization.enabled": True,
            "domain_randomization.rho_min": 0.45,
            "domain_randomization.rho_max": 0.95,
            "domain_randomization.phases": [
                {"rho_min": 0.45, "rho_max": 0.70, "epochs": 20, "horizon": 500},
                {"rho_min": 0.50, "rho_max": 0.85, "epochs": 30, "horizon": 2000},
                {"rho_min": 0.60, "rho_max": 0.95, "epochs": 50, "horizon": 5000},
            ],
        },
    },
    {
        "label": "dr_aggressive",
        "overrides": {
            "domain_randomization.enabled": True,
            "domain_randomization.rho_min": 0.55,
            "domain_randomization.rho_max": 0.98,
            "domain_randomization.phases": [
                {"rho_min": 0.55, "rho_max": 0.78, "epochs": 15, "horizon": 500},
                {"rho_min": 0.60, "rho_max": 0.90, "epochs": 30, "horizon": 2500},
                {"rho_min": 0.70, "rho_max": 0.98, "epochs": 55, "horizon": 7000},
            ],
        },
    },
]


@dataclass
class StudyConfig:
    stage: str = "A"
    mode: str = "full"
    candidate_manifest: str | None = None
    candidate_count: int | None = None
    promote_top_k: int | None = None
    seed_values: list[int] = field(default_factory=list)
    random_seed: int = 20260331
    full_certification: bool = False
    allow_wandb: bool = False
    scorecard: dict[str, Any] = field(default_factory=dict)


def _study_defaults(stage: str) -> dict[str, Any]:
    stage_key = str(stage).upper()
    if stage_key not in STAGE_PRESETS:
        raise ValueError(f"Unknown study stage '{stage}'. Expected one of {sorted(STAGE_PRESETS)}.")
    preset = dict(STAGE_PRESETS[stage_key])
    preset.setdefault("scorecard", {})
    return preset


def normalize_study_config(raw_study_cfg: Any) -> tuple[StudyConfig, dict[str, Any]]:
    payload = {}
    if raw_study_cfg is not None:
        if isinstance(raw_study_cfg, DictConfig):
            payload = OmegaConf.to_container(raw_study_cfg, resolve=True)  # type: ignore[assignment]
        else:
            payload = dict(raw_study_cfg)
    cfg = StudyConfig(**payload)
    preset = _study_defaults(cfg.stage)
    cfg.stage = cfg.stage.upper()
    if cfg.candidate_count is None:
        cfg.candidate_count = int(preset["candidate_count"])
    if cfg.promote_top_k is None:
        cfg.promote_top_k = int(preset["promote_top_k"])
    if not cfg.seed_values:
        cfg.seed_values = [cfg.random_seed + i for i in range(int(preset["seed_count"]))]
    if cfg.mode not in {"full", "train_only", "eval_only"}:
        raise ValueError(f"study.mode must be one of ['full', 'train_only', 'eval_only'], got '{cfg.mode}'")
    if cfg.candidate_count < 1:
        raise ValueError(f"study.candidate_count must be >= 1, got {cfg.candidate_count}")
    if cfg.promote_top_k < 1:
        raise ValueError(f"study.promote_top_k must be >= 1, got {cfg.promote_top_k}")
    return cfg, preset


def build_taxonomy_artifact() -> dict[str, Any]:
    return {
        "frozen_scientific_contract": FROZEN_CONTRACT_PATHS,
        "searchable_scientific_knobs": SEARCHABLE_KNOBS,
        "budget_knobs": BUDGET_KNOBS,
        "bc_data_defaults": DEFAULT_BC_DATA_CONFIG,
        "round_sequence": list(ROUND_SEQUENCE),
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _configs_dir() -> Path:
    return _project_root() / "configs"


def _load_stage_profile_raw(profile_name: str) -> DictConfig:
    raw = OmegaConf.load(_configs_dir() / f"{profile_name}.yaml")
    raw.active_profile = profile_name
    return raw


def _unflatten_mapping(values: dict[str, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path, value in values.items():
        current = root
        parts = path.split(".")
        for key in parts[:-1]:
            current = current.setdefault(key, {})
        current[parts[-1]] = value
    return root


def _candidate_raw(
    base_raw: DictConfig,
    *,
    profile_name: str,
    candidate_id: str,
    seed: int,
    output_root: Path,
    overrides: dict[str, Any],
    allow_wandb: bool = False,
) -> DictConfig:
    wandb_cfg = {
        "enabled": False,
        "mode": "disabled",
        "run_name": f"{candidate_id}-s{seed}",
    }
    if allow_wandb:
        wandb_cfg = {
            "enabled": True,
            "run_name": f"{candidate_id}-s{seed}",
        }
    merged = OmegaConf.merge(
        base_raw,
        OmegaConf.create(_unflatten_mapping(overrides)),
        OmegaConf.create(
            {
                "simulation": {"seed": int(seed)},
                "output_dir": str(output_root),
                "active_profile": profile_name,
                "wandb": wandb_cfg,
            }
        ),
    )
    return OmegaConf.create(OmegaConf.to_container(merged, resolve=True))


def _sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _sample_choice(rng: np.random.Generator, values: list[Any]) -> Any:
    return values[int(rng.integers(0, len(values)))]


def _generate_round_candidate_overrides(
    round_name: str,
    *,
    rng: np.random.Generator,
    parent_overrides: dict[str, Any],
    stage: str,
    refine: bool,
) -> dict[str, Any]:
    candidate = dict(parent_overrides)
    if round_name == "bc":
        if refine:
            candidate["neural_training.bc_lr"] = float(
                np.clip(float(candidate.get("neural_training.bc_lr", 0.002)) * rng.uniform(0.7, 1.3), 5e-4, 5e-3)
            )
            candidate["neural_training.bc_num_steps"] = int(
                np.clip(int(candidate.get("neural_training.bc_num_steps", 1000)) + int(rng.integers(-300, 301)), 100, 4000)
            )
            candidate["bc_data.expert_sim_time"] = float(
                np.clip(float(candidate.get("bc_data.expert_sim_time", 1500.0)) * rng.uniform(0.8, 1.25), 300.0, 5000.0)
            )
        else:
            candidate["neural_training.bc_lr"] = _sample_log_uniform(rng, 5e-4, 5e-3)
            candidate["neural_training.bc_num_steps"] = int(rng.integers(200, 2501))
            candidate["neural_training.bc_label_smoothing"] = float(rng.uniform(0.0, 0.2))
            candidate["bc_data.expert_sim_time"] = float(rng.uniform(600.0, 2500.0))
            candidate["bc_data.rhos"] = _sample_choice(rng, BC_RHO_LIBRARY)
            candidate["bc_data.mu_scales"] = _sample_choice(rng, BC_MU_SCALE_LIBRARY)
            candidate["bc_data.augmentation_noise_min"] = int(rng.integers(-2, 0))
            candidate["bc_data.augmentation_noise_max"] = int(rng.integers(1, 3))
    elif round_name == "reinforce":
        if refine:
            candidate["neural.actor_lr"] = float(
                np.clip(float(candidate.get("neural.actor_lr", 3e-4)) * rng.uniform(0.7, 1.3), 1e-5, 5e-3)
            )
            candidate["neural.critic_lr"] = float(
                np.clip(float(candidate.get("neural.critic_lr", 1e-3)) * rng.uniform(0.7, 1.3), 1e-5, 5e-3)
            )
            candidate["train_epochs"] = int(
                np.clip(int(candidate.get("train_epochs", 30)) + int(rng.integers(-10, 21)), 10, 160)
            )
        else:
            candidate["neural.actor_lr"] = _sample_log_uniform(rng, 1e-4, 3e-3)
            candidate["neural.critic_lr"] = _sample_log_uniform(rng, 3e-4, 3e-3)
            candidate["neural.weight_decay"] = _sample_log_uniform(rng, 1e-5, 5e-3)
            candidate["neural.clip_global_norm"] = float(rng.uniform(0.25, 1.5))
            candidate["neural.lr_decay_rate"] = float(rng.uniform(0.8, 0.99))
            candidate["neural.entropy_bonus"] = float(rng.uniform(0.0, 0.05))
            candidate["neural.entropy_final"] = float(rng.uniform(0.0, 0.01))
            candidate["neural_training.shake_scale"] = float(rng.uniform(0.0, 0.05))
            candidate["neural_training.gae_lambda"] = float(rng.uniform(0.8, 1.0))
            candidate["batch_size"] = int(_sample_choice(rng, [8, 16, 32, 64]))
            candidate["train_epochs"] = int(rng.integers(20, 121))
            candidate["simulation.ssa.sim_time"] = float(rng.uniform(1000.0, 12000.0 if stage != "C" else 25000.0))
    elif round_name == "domain_randomization":
        profile = _sample_choice(rng, DR_LIBRARY)
        candidate.update(profile["overrides"])
        candidate["study.meta.dr_profile"] = profile["label"]
    elif round_name == "architecture":
        if refine:
            candidate["neural.rho_input_scale"] = float(
                np.clip(float(candidate.get("neural.rho_input_scale", 10.0)) * rng.uniform(0.8, 1.2), 0.5, 20.0)
            )
        else:
            candidate["neural.hidden_size"] = int(_sample_choice(rng, [64, 128, 256]))
            candidate["neural.preprocessing"] = _sample_choice(rng, ["none", "log1p", "standardize"])
            candidate["neural.init_type"] = _sample_choice(rng, ["zero_final", "xavier_uniform"])
            candidate["neural.rho_input_scale"] = float(_sample_choice(rng, [1.0, 3.0, 5.0, 10.0]))
            candidate["neural.use_rho"] = bool(_sample_choice(rng, [True, False]))
            candidate["neural.use_service_rates"] = bool(_sample_choice(rng, [True, False]))
    return candidate


def _generate_round_candidates(
    round_name: str,
    *,
    parents: list[dict[str, Any]],
    stage: str,
    candidate_count: int,
    random_seed: int,
    refine: bool,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    candidates: list[dict[str, Any]] = []
    while len(candidates) < candidate_count:
        parent = parents[len(candidates) % len(parents)] if parents else {}
        candidate = _generate_round_candidate_overrides(
            round_name,
            rng=rng,
            parent_overrides=parent,
            stage=stage,
            refine=refine,
        )
        candidates.append(candidate)
    return candidates


def _load_prior_candidates(path: Path, promote_top_k: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Candidate manifest at {path} must be a JSON list.")
    sorted_rows = sorted(payload, key=lambda row: int(row.get("rank", 10_000)))
    parents = [dict(row["overrides"]) for row in sorted_rows[:promote_top_k]]
    if not parents:
        raise ValueError(f"Candidate manifest at {path} did not contain any ranked candidates.")
    return parents


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_reinforce_run(run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(metrics_path(run_dir, "reinforce_metrics.jsonl"))
    if not rows:
        rows = _read_jsonl(run_dir / "reinforce_metrics.jsonl")
    if not rows:
        return {
            "run_dir": str(run_dir),
            "has_nans": True,
            "max_policy_grad_norm": float("inf"),
            "max_value_grad_norm": float("inf"),
            "final_performance_index_ema": float("-inf"),
            "monotone_fraction": 0.0,
            "improved": False,
        }
    perf = np.asarray([float(row.get("performance_index_ema", row.get("performance_index", 0.0))) for row in rows], dtype=np.float64)
    pol_grad = np.asarray([float(row.get("policy_grad_norm", 0.0)) for row in rows], dtype=np.float64)
    val_grad = np.asarray([float(row.get("value_grad_norm", 0.0)) for row in rows], dtype=np.float64)
    diffs = np.diff(perf)
    has_nans = (not np.isfinite(perf).all()) or (not np.isfinite(pol_grad).all()) or (not np.isfinite(val_grad).all())
    monotone_fraction = float(np.mean(diffs >= 0.0)) if diffs.size else 1.0
    return {
        "run_dir": str(run_dir),
        "has_nans": bool(has_nans),
        "max_policy_grad_norm": float(np.max(pol_grad)) if pol_grad.size else 0.0,
        "max_value_grad_norm": float(np.max(val_grad)) if val_grad.size else 0.0,
        "final_performance_index_ema": float(perf[-1]),
        "start_performance_index_ema": float(perf[0]),
        "monotone_fraction": monotone_fraction,
        "improved": bool(perf[-1] > perf[0]),
    }


def summarize_policy_results(results: dict[str, Any]) -> dict[str, Any]:
    neural = results.get("N-GibbsQ (Proposed)") or results.get("N-GibbsQ (Platinum)")
    if neural is None:
        raise KeyError("Policy comparison results are missing the N-GibbsQ row.")
    parity = str(neural.get("parity", "FAILED"))
    jssq = results.get("JSSQ (Min Sojourn)", {})
    uas = results.get("UAS", results.get("UAS (alpha=10.0)", {}))
    calibrated = results.get("Calibrated UAS", {})
    return {
        "parity": parity,
        "parity_numeric": PARITY_ORDER.get(parity, 0),
        "neural_mean_q_total": float(neural["mean_q_total"]),
        "jssq_mean_q_total": float(jssq.get("mean_q_total", float("inf"))),
        "uas_mean_q_total": float(uas.get("mean_q_total", float("inf"))),
        "calibrated_uas_mean_q_total": float(calibrated.get("mean_q_total", float("inf"))),
    }


def summarize_generalization_results(results: dict[str, Any]) -> dict[str, Any]:
    grid = np.asarray(results.get("grid", []), dtype=np.float64)
    return {
        "min_improvement_ratio": float(np.min(grid)) if grid.size else 0.0,
        "mean_improvement_ratio": float(np.mean(grid)) if grid.size else 0.0,
    }


def summarize_critical_results(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_neural_to_gibbs_ratio": float(results.get("max_neural_to_gibbs_ratio", float("inf"))),
        "mean_neural_to_gibbs_ratio": float(results.get("mean_neural_to_gibbs_ratio", float("inf"))),
    }


def summarize_stress_run(run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(metrics_path(run_dir))
    if not rows:
        rows = _read_jsonl(run_dir / "metrics.jsonl")
    critical_rows = [row for row in rows if row.get("test") == "critical_load"]
    hetero_rows = [row for row in rows if row.get("test") == "heterogeneity"]
    stationarity = [float(row.get("stationary_rate", 0.0)) for row in critical_rows]
    hetero_ginis = [float(row.get("gini", 0.0)) for row in hetero_rows]
    return {
        "min_stationarity_rate": min(stationarity) if stationarity else 0.0,
        "mean_stationarity_rate": float(np.mean(stationarity)) if stationarity else 0.0,
        "heterogeneity_gini": hetero_ginis[-1] if hetero_ginis else float("inf"),
    }


def _scorecard_thresholds(study_cfg: StudyConfig) -> dict[str, Any]:
    thresholds = {
        "min_parity": "SILVER",
        "generalization_floor": 0.95,
        "critical_ratio_cap": 1.25,
        "stationarity_rate_min": 1.0,
        "max_grad_norm": 1.0e4,
        "min_monotone_fraction": 0.50,
    }
    thresholds.update(dict(study_cfg.scorecard or {}))
    return thresholds


def compute_trial_scorecard(
    trial: dict[str, Any],
    *,
    study_cfg: StudyConfig,
    suite: list[str],
) -> dict[str, Any]:
    thresholds = _scorecard_thresholds(study_cfg)
    training = trial.get("training", {})
    policy = trial.get("policy", {})
    generalization = trial.get("generalization", {})
    critical = trial.get("critical", {})
    stress = trial.get("stress", {})

    training_required = study_cfg.mode != "eval_only"
    training_pass = (
        True
        if not training_required
        else (
            not training.get("has_nans", True)
            and float(training.get("max_policy_grad_norm", float("inf"))) <= float(thresholds["max_grad_norm"])
            and float(training.get("max_value_grad_norm", float("inf"))) <= float(thresholds["max_grad_norm"])
            and float(training.get("monotone_fraction", 0.0)) >= float(thresholds["min_monotone_fraction"])
            and bool(training.get("improved", False))
        )
    )
    policy_pass = (
        "policy" not in suite
        or int(policy.get("parity_numeric", 0)) >= int(MIN_PARITY_BY_NAME[str(thresholds["min_parity"])])
    )
    generalization_pass = (
        "generalize" not in suite
        or float(generalization.get("min_improvement_ratio", 0.0)) >= float(thresholds["generalization_floor"])
    )
    critical_pass = (
        "critical" not in suite
        or float(critical.get("max_neural_to_gibbs_ratio", float("inf"))) <= float(thresholds["critical_ratio_cap"])
    )
    stress_pass = (
        "stress" not in suite
        or float(stress.get("min_stationarity_rate", 0.0)) >= float(thresholds["stationarity_rate_min"])
    )
    all_pass = training_pass and policy_pass and generalization_pass and critical_pass and stress_pass

    training_score = float(trial.get("training", {}).get("final_performance_index_ema", float("-inf")))
    condition_scores = []
    if "generalize" in suite:
        condition_scores.append(float(generalization.get("min_improvement_ratio", 0.0)))
    if "critical" in suite:
        ratio = float(critical.get("max_neural_to_gibbs_ratio", float("inf")))
        condition_scores.append(0.0 if not np.isfinite(ratio) else 1.0 / max(ratio, 1e-6))
    if "stress" in suite:
        condition_scores.append(float(stress.get("min_stationarity_rate", 0.0)))
    if "policy" in suite:
        condition_scores.append(float(policy.get("parity_numeric", 0)))
    eval_seed_score = float(np.mean(condition_scores)) if condition_scores else float("-inf")
    worst_seed_score = training_score if np.isfinite(training_score) else eval_seed_score
    worst_condition_score = min(condition_scores) if condition_scores else worst_seed_score
    composite = (
        float(policy.get("parity_numeric", 0))
        + float(generalization.get("mean_improvement_ratio", 0.0))
        + (
            0.0
            if not np.isfinite(float(critical.get("mean_neural_to_gibbs_ratio", float("inf"))))
            else 1.0 / max(float(critical.get("mean_neural_to_gibbs_ratio", float("inf"))), 1e-6)
        )
        + float(training.get("final_performance_index_ema", 0.0)) / 100.0
    )

    return {
        "all_gates_pass": all_pass,
        "training_pass": training_pass,
        "policy_pass": policy_pass,
        "generalization_pass": generalization_pass,
        "critical_pass": critical_pass,
        "stress_pass": stress_pass,
        "worst_seed_performance": worst_seed_score,
        "worst_condition_score": worst_condition_score,
        "composite_score": composite,
    }


def aggregate_candidate_trials(
    *,
    candidate_id: str,
    overrides: dict[str, Any],
    trials: list[dict[str, Any]],
    study_cfg: StudyConfig,
    suite: list[str],
) -> dict[str, Any]:
    scorecards = [compute_trial_scorecard(trial, study_cfg=study_cfg, suite=suite) for trial in trials]
    gate_passes = [bool(card["all_gates_pass"]) for card in scorecards]
    worst_seed = min(float(card["worst_seed_performance"]) for card in scorecards)
    worst_condition = min(float(card["worst_condition_score"]) for card in scorecards)
    composite_scores = [float(card["composite_score"]) for card in scorecards]
    parity_scores = [int(trial.get("policy", {}).get("parity_numeric", 0)) for trial in trials if "policy" in trial]
    generalization_scores = [float(trial.get("generalization", {}).get("mean_improvement_ratio", 0.0)) for trial in trials if "generalization" in trial]
    return {
        "candidate_id": candidate_id,
        "overrides": overrides,
        "seed_values": [int(trial["seed"]) for trial in trials],
        "trials": trials,
        "passes_all_gates": all(gate_passes) if gate_passes else False,
        "worst_seed_performance": worst_seed,
        "worst_condition_score": worst_condition,
        "median_parity_score": float(statistics.median(parity_scores)) if parity_scores else 0.0,
        "median_generalization_score": float(statistics.median(generalization_scores)) if generalization_scores else 0.0,
        "mean_score": float(np.mean(composite_scores)) if composite_scores else float("-inf"),
    }


def rank_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row["passes_all_gates"] else 1,
            -float(row["worst_seed_performance"]),
            -float(row["worst_condition_score"]),
            -float(row["median_parity_score"] + row["median_generalization_score"]),
            -float(row["mean_score"]),
            row["candidate_id"],
        ),
    )
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def _latest_run_dir(base_output_dir: Path, experiment_name: str) -> Path:
    exp_dir = base_output_dir / experiment_name
    candidates = [path for path in exp_dir.iterdir() if path.is_dir()] if exp_dir.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {exp_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_reinforce_trial(raw_candidate: DictConfig, profile_name: str) -> dict[str, Any]:
    from experiments.training.train_reinforce import ReinforceTrainer

    cfg, resolved_raw = load_experiment_config(raw_candidate, "reinforce_train", profile_name=profile_name)
    run_dir, _ = get_run_config(cfg, "reinforce_train", resolved_raw)
    trainer = ReinforceTrainer(cfg, run_dir, run_logger=None, bc_data_config=extract_bc_data_config(resolved_raw))
    trainer.execute(jax.random.PRNGKey(cfg.simulation.seed), n_epochs=cfg.train_epochs)
    return summarize_reinforce_run(run_dir)


def _run_policy_trial(raw_candidate: DictConfig, profile_name: str) -> dict[str, Any]:
    from experiments.evaluation.baselines_comparison import run_corrected_comparison

    cfg, resolved_raw = load_experiment_config(raw_candidate, "policy", profile_name=profile_name)
    run_dir, _ = get_run_config(cfg, "policy", resolved_raw)
    results = run_corrected_comparison(cfg, run_dir, run_logger=None)
    summary = summarize_policy_results(results)
    summary["run_dir"] = str(run_dir)
    return summary


def _run_generalization_trial(raw_candidate: DictConfig, profile_name: str) -> dict[str, Any]:
    from experiments.evaluation.n_gibbsq_evals.gen_sweep import GeneralizationSweeper

    cfg, resolved_raw = load_experiment_config(raw_candidate, "generalize", profile_name=profile_name)
    run_dir, _ = get_run_config(cfg, "generalize", resolved_raw)
    results = GeneralizationSweeper(cfg, run_dir, run_logger=None).execute(jax.random.PRNGKey(cfg.simulation.seed))
    summary = summarize_generalization_results(results)
    summary["run_dir"] = str(run_dir)
    return summary


def _run_critical_trial(raw_candidate: DictConfig, profile_name: str) -> dict[str, Any]:
    from experiments.evaluation.n_gibbsq_evals.critical_load import CriticalLoadTest

    cfg, resolved_raw = load_experiment_config(raw_candidate, "critical", profile_name=profile_name)
    run_dir, _ = get_run_config(cfg, "critical", resolved_raw)
    results = CriticalLoadTest(cfg, run_dir, run_logger=None).execute(jax.random.PRNGKey(cfg.simulation.seed))
    summary = summarize_critical_results(results)
    summary["run_dir"] = str(run_dir)
    return summary


def _run_stress_trial(raw_candidate: DictConfig, profile_name: str) -> dict[str, Any]:
    from experiments.testing import stress_test as stress_module

    cfg, _ = load_experiment_config(raw_candidate, "stress", profile_name=profile_name)
    if not cfg.jax.enabled:
        return {
            "run_dir": "",
            "min_stationarity_rate": 0.0,
            "mean_stationarity_rate": 0.0,
            "heterogeneity_gini": float("inf"),
        }
    stress_module.main(raw_candidate)
    base_output = Path(str(OmegaConf.select(raw_candidate, "output_dir")))
    run_dir = _latest_run_dir(base_output, "stress")
    summary = summarize_stress_run(run_dir)
    summary["run_dir"] = str(run_dir)
    return summary


def execute_seed_trial(
    *,
    raw_candidate: DictConfig,
    profile_name: str,
    suite: list[str],
    mode: str,
) -> dict[str, Any]:
    seed = int(OmegaConf.select(raw_candidate, "simulation.seed"))
    result: dict[str, Any] = {"seed": seed}
    if mode in {"full", "train_only"}:
        result["training"] = _run_reinforce_trial(raw_candidate, profile_name)
    if mode in {"full", "eval_only"}:
        if "policy" in suite:
            result["policy"] = _run_policy_trial(raw_candidate, profile_name)
        if "generalize" in suite:
            result["generalization"] = _run_generalization_trial(raw_candidate, profile_name)
        if "critical" in suite:
            result["critical"] = _run_critical_trial(raw_candidate, profile_name)
        if "stress" in suite:
            result["stress"] = _run_stress_trial(raw_candidate, profile_name)
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_seed_dispersion(path: Path, ranked_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, row in enumerate(ranked_rows, start=1):
        ys = [float(trial.get("training", {}).get("final_performance_index_ema", float("nan"))) for trial in row["trials"]]
        xs = np.full(len(ys), idx)
        ax.scatter(xs, ys, alpha=0.7, label=row["candidate_id"] if idx <= 5 else None)
    ax.set_xlabel("Candidate Rank")
    ax.set_ylabel("Final EMA Performance Index")
    ax.set_title("Seed Dispersion")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_pareto(path: Path, ranked_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [float(row["worst_condition_score"]) for row in ranked_rows]
    ys = [float(row["mean_score"]) for row in ranked_rows]
    colors = ["green" if row["passes_all_gates"] else "red" for row in ranked_rows]
    ax.scatter(xs, ys, c=colors, alpha=0.8)
    for row, x, y in zip(ranked_rows[:8], xs[:8], ys[:8]):
        ax.annotate(row["candidate_id"], (x, y), fontsize=8)
    ax.set_xlabel("Worst-Condition Score")
    ax.set_ylabel("Mean Composite Score")
    ax.set_title("Pareto View: Robustness vs Mean Score")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_recommendation(path: Path, *, stage: str, ranked_rows: list[dict[str, Any]], suite: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Hyperparameter Qualification Recommendation ({stage})",
        "",
        f"Evaluation suite: {', '.join(suite) if suite else 'training-only'}",
        "",
    ]
    if ranked_rows:
        winner = ranked_rows[0]
        lines.extend(
            [
                "## Selected Candidate",
                "",
                f"- Candidate: `{winner['candidate_id']}`",
                f"- Passes all gates: `{winner['passes_all_gates']}`",
                f"- Worst-seed performance: `{winner['worst_seed_performance']:.4f}`",
                f"- Worst-condition score: `{winner['worst_condition_score']:.4f}`",
                "",
                "## Overrides",
                "",
                "```json",
                json.dumps(winner["overrides"], indent=2),
                "```",
                "",
                "## Near Misses",
                "",
            ]
        )
        for row in ranked_rows[1:4]:
            lines.append(
                f"- `{row['candidate_id']}`: gates={row['passes_all_gates']} "
                f"| worst-seed={row['worst_seed_performance']:.4f} "
                f"| worst-condition={row['worst_condition_score']:.4f}"
            )
    else:
        lines.append("No candidates were evaluated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage(
    *,
    stage: str,
    study_cfg: StudyConfig,
    preset: dict[str, Any],
    hyperqual_run_dir: Path,
) -> dict[str, Any]:
    profile_name = str(preset["profile_name"])
    suite = list(preset["suite"])
    if study_cfg.full_certification:
        suite = ["policy", "generalize", "critical", "stress"]
    base_raw = _load_stage_profile_raw(profile_name)
    round_output = hyperqual_run_dir / f"stage_{stage.lower()}"
    _write_json(round_output / "taxonomy.json", build_taxonomy_artifact())

    if stage == "A":
        parents = [{}]
    else:
        if not study_cfg.candidate_manifest:
            raise ValueError(f"Stage {stage} requires study.candidate_manifest from the previous stage.")
        manifest_path = Path(study_cfg.candidate_manifest)
        if not manifest_path.is_absolute():
            manifest_path = (_project_root() / manifest_path).resolve()
        parents = _load_prior_candidates(manifest_path, study_cfg.promote_top_k)

    stage_leaderboard: list[dict[str, Any]] = []
    current_parents = parents
    round_names = list(ROUND_SEQUENCE) if stage in {"A", "B"} else ["certification"]
    for round_index, round_name in enumerate(round_names, start=1):
        refine = stage == "B"
        if round_name == "certification":
            candidate_overrides_list = current_parents[: study_cfg.candidate_count]
        else:
            candidate_overrides_list = _generate_round_candidates(
                round_name,
                parents=current_parents,
                stage=stage,
                candidate_count=study_cfg.candidate_count,
                random_seed=study_cfg.random_seed + round_index * 1000,
                refine=refine,
            )

        aggregated_rows: list[dict[str, Any]] = []
        for candidate_index, overrides in enumerate(candidate_overrides_list, start=1):
            candidate_id = f"{stage.lower()}-{round_name[:3]}-{candidate_index:03d}"
            candidate_output_root = round_output / "candidates" / candidate_id
            trial_rows: list[dict[str, Any]] = []
            for seed in study_cfg.seed_values:
                raw_candidate = _candidate_raw(
                    base_raw,
                    profile_name=profile_name,
                    candidate_id=candidate_id,
                    seed=seed,
                    output_root=candidate_output_root,
                    overrides=overrides,
                    allow_wandb=study_cfg.allow_wandb,
                )
                trial = execute_seed_trial(
                    raw_candidate=raw_candidate,
                    profile_name=profile_name,
                    suite=[] if study_cfg.mode == "train_only" else suite,
                    mode=study_cfg.mode,
                )
                trial_manifest = {
                    "candidate_id": candidate_id,
                    "stage": stage,
                    "round": round_name,
                    "profile_name": profile_name,
                    "seed": seed,
                    "suite": suite,
                    "mode": study_cfg.mode,
                    "overrides": overrides,
                    "bc_data": extract_bc_data_config(raw_candidate),
                    "result": trial,
                }
                _write_json(
                    round_output / "trial_manifests" / f"{candidate_id}__seed{seed}.json",
                    trial_manifest,
                )
                trial_rows.append(trial)

            aggregated = aggregate_candidate_trials(
                candidate_id=candidate_id,
                overrides=overrides,
                trials=trial_rows,
                study_cfg=study_cfg,
                suite=suite,
            )
            aggregated_rows.append(aggregated)

        ranked = rank_candidates(aggregated_rows)
        stage_leaderboard = ranked
        leaderboard_export = [
            {
                "rank": row["rank"],
                "candidate_id": row["candidate_id"],
                "passes_all_gates": row["passes_all_gates"],
                "worst_seed_performance": row["worst_seed_performance"],
                "worst_condition_score": row["worst_condition_score"],
                "median_parity_score": row["median_parity_score"],
                "median_generalization_score": row["median_generalization_score"],
                "mean_score": row["mean_score"],
                "overrides": row["overrides"],
            }
            for row in ranked
        ]
        _write_json(round_output / f"{round_name}_leaderboard.json", leaderboard_export)
        _write_csv(
            round_output / f"{round_name}_leaderboard.csv",
            [
                {
                    "rank": row["rank"],
                    "candidate_id": row["candidate_id"],
                    "passes_all_gates": row["passes_all_gates"],
                    "worst_seed_performance": row["worst_seed_performance"],
                    "worst_condition_score": row["worst_condition_score"],
                    "median_parity_score": row["median_parity_score"],
                    "median_generalization_score": row["median_generalization_score"],
                    "mean_score": row["mean_score"],
                    "overrides_json": json.dumps(row["overrides"], sort_keys=True),
                }
                for row in ranked
            ],
            [
                "rank",
                "candidate_id",
                "passes_all_gates",
                "worst_seed_performance",
                "worst_condition_score",
                "median_parity_score",
                "median_generalization_score",
                "mean_score",
                "overrides_json",
            ],
        )
        _plot_seed_dispersion(round_output / f"{round_name}_seed_dispersion.png", ranked)
        _plot_pareto(round_output / f"{round_name}_pareto.png", ranked)

        current_parents = [dict(row["overrides"]) for row in ranked[: study_cfg.promote_top_k]]
        if stage == "C":
            break

    _write_recommendation(round_output / "final_recommendation.md", stage=stage, ranked_rows=stage_leaderboard, suite=suite)
    return {
        "stage": stage,
        "profile_name": profile_name,
        "suite": suite,
        "leaderboard": stage_leaderboard,
        "output_dir": str(round_output),
    }
