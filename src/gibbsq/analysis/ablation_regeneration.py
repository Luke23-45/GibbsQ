"""Regenerate premium ablation figures from saved run artifacts."""

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

from gibbsq.analysis.plotting import plot_ablation_dual_panel
from gibbsq.core.config import ExperimentConfig
from gibbsq.core.neural_policies import NeuralRouter
from gibbsq.utils.chart_exporter import save_data
from gibbsq.utils.run_artifacts import artifacts_dir, config_path, figure_path, metrics_path

CANONICAL_VARIANTS = [
    "Full Model",
    "Ablated: No Log-Norm",
    "Ablated: No Zero-Init",
    "Uniform Routing (Baseline)",
]

VARIANT_SPECS = [
    {
        "variant": "Full Model",
        "preprocessing": "log1p",
        "init_type": "zero_final",
        "artifact_dir": "variant_1_full_model",
    },
    {
        "variant": "Ablated: No Log-Norm",
        "preprocessing": "none",
        "init_type": "zero_final",
        "artifact_dir": "variant_2_ablated_no_log-norm",
    },
    {
        "variant": "Ablated: No Zero-Init",
        "preprocessing": "log1p",
        "init_type": "standard",
        "artifact_dir": "variant_3_ablated_no_zero-init",
    },
]


@dataclass(frozen=True)
class AblationRecord:
    variant: str
    preprocessing: str
    init_type: str
    mean_q_total: float
    se_q_total: float


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


def load_validated_ablation_records(path: Path) -> list[AblationRecord]:
    rows = _load_jsonl_records(path)
    if len(rows) != len(CANONICAL_VARIANTS):
        raise ValueError(
            f"Expected {len(CANONICAL_VARIANTS)} ablation rows in {path}, found {len(rows)}"
        )

    seen: set[str] = set()
    validated: list[AblationRecord] = []
    for idx, (payload, expected_variant) in enumerate(zip(rows, CANONICAL_VARIANTS), start=1):
        variant = payload.get("variant")
        if variant != expected_variant:
            raise ValueError(
                f"Unexpected variant order in {path} at row {idx}: "
                f"expected '{expected_variant}', found '{variant}'"
            )
        if variant in seen:
            raise ValueError(f"Duplicate variant '{variant}' in {path}")
        seen.add(variant)

        preprocessing = payload.get("preprocessing")
        init_type = payload.get("init_type")
        if not isinstance(preprocessing, str) or not isinstance(init_type, str):
            raise ValueError(f"Variant '{variant}' is missing preprocessing/init_type metadata")

        try:
            mean_q_total = float(payload["mean_q_total"])
            se_q_total = float(payload["se_q_total"])
        except KeyError as exc:
            raise ValueError(f"Variant '{variant}' is missing metric '{exc.args[0]}'") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Variant '{variant}' has non-numeric metrics") from exc

        if not np.isfinite(mean_q_total) or not np.isfinite(se_q_total):
            raise ValueError(f"Variant '{variant}' has non-finite metrics")

        validated.append(
            AblationRecord(
                variant=variant,
                preprocessing=preprocessing,
                init_type=init_type,
                mean_q_total=mean_q_total,
                se_q_total=se_q_total,
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
    from experiments.evaluation.n_gibbsq_evals.ablation_ssa import (
        UniformRouting,
        _build_ablation_eval_policy,
        evaluate_policy_ssa,
    )

    cfg = _load_experiment_config(run_dir)
    reconstructed: list[AblationRecord] = []

    for idx, spec in enumerate(VARIANT_SPECS):
        variant_dir = artifacts_dir(run_dir) / spec["artifact_dir"]
        model_path = artifacts_dir(variant_dir) / "n_gibbsq_reinforce_weights.eqx"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for fallback regeneration: {model_path}")

        neural_cfg = cfg.neural
        if spec["preprocessing"] != neural_cfg.preprocessing or spec["init_type"] != neural_cfg.init_type:
            neural_cfg = dataclasses.replace(
                neural_cfg,
                preprocessing=spec["preprocessing"],
                init_type=spec["init_type"],
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
                variant=spec["variant"],
                preprocessing=spec["preprocessing"],
                init_type=spec["init_type"],
                mean_q_total=metrics["mean_q_total"],
                se_q_total=metrics["se_q_total"],
            )
        )

    baseline = evaluate_policy_ssa(UniformRouting(), cfg)
    reconstructed.append(
        AblationRecord(
            variant="Uniform Routing (Baseline)",
            preprocessing="n/a",
            init_type="n/a",
            mean_q_total=baseline["mean_q_total"],
            se_q_total=baseline["se_q_total"],
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

    full_model = ordered[0].mean_q_total
    return {
        "variants": [record.variant for record in ordered],
        "preprocessing": [record.preprocessing for record in ordered],
        "init_type": [record.init_type for record in ordered],
        "mean_q_total": [record.mean_q_total for record in ordered],
        "se_q_total": [record.se_q_total for record in ordered],
        "delta_pct_vs_full_model": [
            ((record.mean_q_total - full_model) / full_model) * 100.0
            for record in ordered
        ],
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

    plot_ablation_dual_panel(
        variant_names=payload["variants"],
        mean_values=payload["mean_q_total"],
        se_values=payload["se_q_total"],
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
