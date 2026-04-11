"""Regenerate empirical ablation figures from saved run artifacts."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import equinox as eqx
import jax
import numpy as np

from gibbsq.analysis.plotting import plot_ablation_bars
from gibbsq.core.config import ExperimentConfig
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.utils.chart_exporter import save_data
from gibbsq.utils.run_artifacts import artifacts_dir, config_path, figure_path, metrics_path

from experiments.evaluation.n_gibbsq_evals.ablation_ssa import (
    ALL_VARIANTS,
    NEURAL_VARIANTS,
    REFERENCE_VARIANTS,
    _build_ablation_eval_policy,
    _reference_policy,
    evaluate_policy_ssa,
)


@dataclass(frozen=True)
class AblationRecord:
    variant: str
    variant_kind: str
    panel: str
    preprocessing: str
    init_type: str
    bootstrap_mode: str
    teacher_policy: str
    mean_q_total: float
    se_q_total: float
    ci95_half_width: float
    delta_vs_calibrated_uas_mean: float | None = None
    delta_vs_calibrated_uas_se: float | None = None
    delta_vs_calibrated_uas_ci95_half_width: float | None = None
    delta_vs_best_neural_mean: float | None = None
    delta_vs_best_neural_se: float | None = None
    delta_vs_best_neural_ci95_half_width: float | None = None


CANONICAL_VARIANTS = [spec.name for spec in ALL_VARIANTS]


def _load_jsonl_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL in {path} at line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_no}")
            rows.append(payload)
    return rows


def _optional_float(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        cast_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{key}' must be numeric when present.") from exc
    if not np.isfinite(cast_value):
        raise ValueError(f"Field '{key}' must be finite when present.")
    return cast_value


def _optional_float_compat(payload: dict, primary_key: str, legacy_key: str) -> float | None:
    value = _optional_float(payload, primary_key)
    if value is not None:
        return value
    return _optional_float(payload, legacy_key)


def load_validated_ablation_records(path: Path) -> list[AblationRecord]:
    rows = _load_jsonl_records(path)
    if len(rows) != len(CANONICAL_VARIANTS):
        raise ValueError(
            f"Expected {len(CANONICAL_VARIANTS)} ablation rows in {path}, found {len(rows)}"
        )

    expected_set = set(CANONICAL_VARIANTS)
    found_set = {row.get("variant") for row in rows}
    if found_set != expected_set:
        missing = sorted(expected_set - found_set)
        unexpected = sorted(found_set - expected_set)
        raise ValueError(
            f"Ablation variants in {path} do not match expected schema. "
            f"Missing={missing}, Unexpected={unexpected}"
        )

    row_map = {row["variant"]: row for row in rows}
    ordered_rows = [row_map[name] for name in CANONICAL_VARIANTS]

    validated: list[AblationRecord] = []
    for payload in ordered_rows:
        variant = payload.get("variant")
        if not isinstance(variant, str):
            raise ValueError("Each ablation row must define a string 'variant'.")
        for key in ("variant_kind", "panel", "preprocessing", "init_type", "bootstrap_mode", "teacher_policy"):
            if not isinstance(payload.get(key), str):
                raise ValueError(f"Variant '{variant}' is missing string field '{key}'.")
        try:
            mean_q_total = float(payload["mean_q_total"])
            se_q_total = float(payload["se_q_total"])
            ci95_half_width = float(payload["ci95_half_width"])
        except KeyError as exc:
            raise ValueError(f"Variant '{variant}' is missing metric '{exc.args[0]}'") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Variant '{variant}' has non-numeric primary metrics") from exc
        if not all(np.isfinite(x) for x in (mean_q_total, se_q_total, ci95_half_width)):
            raise ValueError(f"Variant '{variant}' has non-finite primary metrics")

        validated.append(
            AblationRecord(
                variant=variant,
                variant_kind=payload["variant_kind"],
                panel=payload["panel"],
                preprocessing=payload["preprocessing"],
                init_type=payload["init_type"],
                bootstrap_mode=payload["bootstrap_mode"],
                teacher_policy=payload["teacher_policy"],
                mean_q_total=mean_q_total,
                se_q_total=se_q_total,
                ci95_half_width=ci95_half_width,
                delta_vs_calibrated_uas_mean=_optional_float_compat(payload, "delta_vs_calibrated_uas_mean", "delta_vs_refined_uas_mean"),
                delta_vs_calibrated_uas_se=_optional_float_compat(payload, "delta_vs_calibrated_uas_se", "delta_vs_refined_uas_se"),
                delta_vs_calibrated_uas_ci95_half_width=_optional_float_compat(payload, "delta_vs_calibrated_uas_ci95_half_width", "delta_vs_refined_uas_ci95_half_width"),
                delta_vs_best_neural_mean=_optional_float(payload, "delta_vs_best_neural_mean"),
                delta_vs_best_neural_se=_optional_float(payload, "delta_vs_best_neural_se"),
                delta_vs_best_neural_ci95_half_width=_optional_float(payload, "delta_vs_best_neural_ci95_half_width"),
            )
        )

    return validated


def _load_experiment_config(run_dir: Path) -> ExperimentConfig:
    from omegaconf import OmegaConf
    from gibbsq.core.config import validate

    cfg_path = config_path(run_dir)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config for ablation regeneration: {cfg_path}")
    cfg = OmegaConf.to_object(OmegaConf.load(cfg_path))
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError(f"Expected ExperimentConfig from {cfg_path}, found {type(cfg)!r}")
    validate(cfg)
    return cfg


def _reconstruct_records_from_artifacts(run_dir: Path) -> list[AblationRecord]:
    cfg = _load_experiment_config(run_dir)
    reconstructed: list[AblationRecord] = []

    for idx, spec in enumerate(NEURAL_VARIANTS):
        variant_dir = artifacts_dir(run_dir) / (spec.artifact_dir or f"variant_{idx + 1}")
        model_path = artifacts_dir(variant_dir) / "n_gibbsq_reinforce_weights.eqx"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for fallback regeneration: {model_path}")

        neural_cfg = dataclasses.replace(
            cfg.neural,
            preprocessing=spec.preprocessing or cfg.neural.preprocessing,
            init_type=spec.init_type or cfg.neural.init_type,
        )
        variant_cfg = dataclasses.replace(cfg, neural=neural_cfg)

        skeleton = NeuralRouter(
            num_servers=variant_cfg.system.num_servers,
            config=variant_cfg.neural,
            service_rates=variant_cfg.system.service_rates,
            key=jax.random.PRNGKey(variant_cfg.simulation.seed + 10_000 + idx),
        )
        model = eqx.tree_deserialise_leaves(model_path, skeleton)
        mu_arr = np.array(variant_cfg.system.service_rates, dtype=np.float64)
        rho = variant_cfg.system.arrival_rate / float(mu_arr.sum())
        policy = _build_ablation_eval_policy(model, mu_arr, rho)
        metrics = evaluate_policy_ssa(policy, variant_cfg)
        reconstructed.append(
            AblationRecord(
                variant=spec.name,
                variant_kind=spec.variant_kind,
                panel=spec.panel,
                preprocessing=spec.preprocessing or "n/a",
                init_type=spec.init_type or "n/a",
                bootstrap_mode=spec.bootstrap_mode,
                teacher_policy=spec.expert_policy_name or "n/a",
                mean_q_total=metrics["mean_q_total"],
                se_q_total=metrics["se_q_total"],
                ci95_half_width=metrics["ci95_half_width"],
            )
        )

    for spec in REFERENCE_VARIANTS:
        metrics = evaluate_policy_ssa(_reference_policy(spec, cfg), cfg)
        reconstructed.append(
            AblationRecord(
                variant=spec.name,
                variant_kind=spec.variant_kind,
                panel=spec.panel,
                preprocessing="n/a",
                init_type="n/a",
                bootstrap_mode=spec.bootstrap_mode,
                teacher_policy="n/a",
                mean_q_total=metrics["mean_q_total"],
                se_q_total=metrics["se_q_total"],
                ci95_half_width=metrics["ci95_half_width"],
            )
        )

    return reconstructed


def load_ablation_records(
    run_dir: Path,
    *,
    allow_fallback_reconstruction: bool = True,
) -> tuple[list[AblationRecord], str]:
    summary_path = metrics_path(run_dir, "ablation_ssa_metrics.jsonl")
    if not summary_path.exists():
        if not allow_fallback_reconstruction:
            raise FileNotFoundError(f"Missing ablation summary metrics: {summary_path}")
        return _reconstruct_records_from_artifacts(run_dir), "reconstructed_from_artifacts"

    try:
        return load_validated_ablation_records(summary_path), "summary_jsonl"
    except ValueError:
        if not allow_fallback_reconstruction:
            raise
        return _reconstruct_records_from_artifacts(run_dir), "reconstructed_from_artifacts"


def build_ablation_plot_payload(records: Iterable[AblationRecord]) -> dict:
    ordered = list(records)
    if not ordered:
        raise ValueError("Ablation plot payload requires at least one record")

    return {
        "variants": [record.variant for record in ordered],
        "variant_kind": [record.variant_kind for record in ordered],
        "panel": [record.panel for record in ordered],
        "preprocessing": [record.preprocessing for record in ordered],
        "init_type": [record.init_type for record in ordered],
        "bootstrap_mode": [record.bootstrap_mode for record in ordered],
        "teacher_policy": [record.teacher_policy for record in ordered],
        "mean_q_total": [record.mean_q_total for record in ordered],
        "se_q_total": [record.se_q_total for record in ordered],
        "ci95_half_width": [record.ci95_half_width for record in ordered],
        "delta_vs_calibrated_uas_mean": [record.delta_vs_calibrated_uas_mean for record in ordered],
        "delta_vs_best_neural_mean": [record.delta_vs_best_neural_mean for record in ordered],
    }


def regenerate_ablation_figure(
    run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    theme: str = "publication",
    allow_fallback_reconstruction: bool = True,
) -> dict[str, Path | str]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Ablation run directory does not exist: {run_dir}")

    records, source = load_ablation_records(
        run_dir,
        allow_fallback_reconstruction=allow_fallback_reconstruction,
    )
    payload = build_ablation_plot_payload(records)

    target_dir = Path(output_dir) if output_dir else run_dir
    figures_target = figure_path(target_dir, "ablation_ssa_regenerated")
    data_target = metrics_path(target_dir, "ablation_ssa_regenerated_data")
    metadata_target = metrics_path(target_dir, "ablation_ssa_regenerated_metadata")

    plot_ablation_bars(
        variant_names=payload["variants"],
        mean_values=payload["mean_q_total"],
        se_values=payload["ci95_half_width"],
        save_path=figures_target,
        theme=theme,
        formats=["png", "pdf"],
    )

    save_data(payload, data_target, format="json")
    save_data(
        {
            "source_run_dir": str(run_dir.resolve()),
            "figure_stem": figures_target.name,
            "data_source": source,
            "generated_at": datetime.now().astimezone().isoformat(),
            "theme": theme,
        },
        metadata_target,
        format="json",
    )

    return {
        "figure_png": figures_target.with_suffix(".png"),
        "figure_pdf": figures_target.with_suffix(".pdf"),
        "data_json": data_target.with_suffix(".json"),
        "metadata_json": metadata_target.with_suffix(".json"),
        "data_source": source,
    }
