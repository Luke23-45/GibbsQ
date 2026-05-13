"""Audit helpers for experiment run capsules and expected artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ExperimentArtifactSpec:
    """Required and optional artifact expectations for one experiment family."""

    required_figures: tuple[str, ...] = ()
    required_metrics: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    completion_markers: tuple[str, ...] = ()
    optional_figures: tuple[str, ...] = ()
    optional_figure_gate: Callable[[Path], bool] | None = None
    validator: Callable[[Path], tuple[str, ...]] | None = None


@dataclass(frozen=True)
class RunAuditResult:
    """Outcome of auditing one run capsule."""

    experiment: str
    run_dir: Path
    missing_figures: tuple[str, ...]
    missing_metrics: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    missing_completion_markers: tuple[str, ...]
    expected_optional_figures: tuple[str, ...]
    integrity_issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            not self.missing_figures
            and not self.missing_metrics
            and not self.missing_artifacts
            and not self.missing_completion_markers
            and not self.integrity_issues
        )


def _logs_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "logs"


def _figures_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "figures"


def _metrics_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "metrics"


def _config_path(run_dir: Path) -> Path:
    return Path(run_dir) / "metadata" / "config.yaml"


def _config_flag_enabled(run_dir: Path, key: str) -> bool:
    cfg_path = _config_path(run_dir)
    if not cfg_path.exists():
        return False
    cfg = OmegaConf.load(cfg_path)
    value = OmegaConf.select(cfg, key)
    return bool(value)


def _policy_grid_enabled(run_dir: Path) -> bool:
    return _config_flag_enabled(run_dir, "grid")


def _policy_metrics_include_tier5(run_dir: Path) -> tuple[str, ...]:
    metrics_file = _metrics_dir(run_dir) / "corrected_comparison_metrics.jsonl"
    if not metrics_file.exists():
        return ()

    found_tier5 = False
    found_name = False
    with metrics_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            found_tier5 = found_tier5 or row.get("tier") == 5
            found_name = found_name or row.get("policy") in ("N-GibbsQ (Platinum)", "N-GibbsQ (Proposed)")
    issues = []
    if not found_tier5:
        issues.append("policy metrics missing tier=5 row")
    if not found_name:
        issues.append("policy metrics missing N-GibbsQ (Proposed) row")
    return tuple(issues)


def _reinforce_stage_profile_has_final_eval(run_dir: Path) -> tuple[str, ...]:
    profile_path = _metrics_dir(run_dir) / "reinforce_stage_profile.json"
    if not profile_path.exists():
        return ()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    final_eval = profile.get("final_eval")
    if not isinstance(final_eval, dict):
        return ("reinforce stage profile missing final_eval summary",)
    required_keys = {"mean_score", "std_score", "num_scores"}
    missing = required_keys.difference(final_eval)
    if missing:
        return (f"reinforce final_eval missing keys: {', '.join(sorted(missing))}",)
    return ()


EXPERIMENT_ARTIFACT_SPECS: dict[str, ExperimentArtifactSpec] = {
    "bc_train": ExperimentArtifactSpec(
        required_artifacts=("n_gibbsq_platinum_bc_weights.eqx", "n_gibbsq_platinum_bc_weights.eqx.bc_metadata.json"),
        completion_markers=("Platinum BC Weights saved",),
    ),
    "critical": ExperimentArtifactSpec(
        required_figures=("critical_load_curve.png", "critical_load_curve.pdf"),
        required_metrics=("metrics.jsonl",),
        completion_markers=("Critical load test complete. Curve saved",),
    ),
    "drift": ExperimentArtifactSpec(
        required_figures=("drift_heatmap.png", "drift_heatmap.pdf", "drift_vs_norm.png", "drift_vs_norm.pdf"),
        required_metrics=("metrics.jsonl",),
        completion_markers=("Drift verification complete",),
    ),
    "generalize": ExperimentArtifactSpec(
        required_figures=("generalization_heatmap.png", "generalization_heatmap.pdf"),
        required_metrics=("metrics.jsonl",),
        completion_markers=("Generalization analysis complete. Heatmap saved",),
    ),
    "policy": ExperimentArtifactSpec(
        required_figures=("corrected_policy_comparison.png", "corrected_policy_comparison.pdf"),
        required_metrics=("corrected_comparison_metrics.jsonl",),
        completion_markers=("Comparison plot saved",),
        optional_figures=("platinum_grid_analysis.png", "platinum_grid_analysis.pdf"),
        optional_figure_gate=_policy_grid_enabled,
        validator=_policy_metrics_include_tier5,
    ),
    "reinforce_check": ExperimentArtifactSpec(
        required_figures=("gradient_scatter.png", "gradient_scatter.pdf"),
        required_metrics=("gradient_check_result.json",),
        completion_markers=("Gradient scatter plot saved",),
    ),
    "reinforce_train": ExperimentArtifactSpec(
        required_figures=("reinforce_training_curve.png", "reinforce_training_curve.pdf"),
        required_metrics=("reinforce_metrics.jsonl", "reinforce_stage_profile.json"),
        completion_markers=("Stage profile written", "Training Complete!"),
        validator=_reinforce_stage_profile_has_final_eval,
    ),
    "stats": ExperimentArtifactSpec(
        required_figures=("stats_boxplot.png", "stats_boxplot.pdf"),
        required_metrics=("metrics.jsonl",),
        completion_markers=("STATISTICAL SUMMARY",),
    ),
    "stress": ExperimentArtifactSpec(
        required_figures=("stress_dashboard.png", "stress_dashboard.pdf"),
        required_metrics=("metrics.jsonl",),
        completion_markers=("Stress dashboard saved",),
    ),
    "sweep": ExperimentArtifactSpec(
        required_figures=("alpha_sweep.png", "alpha_sweep.pdf"),
        required_metrics=("metrics.jsonl",),
        completion_markers=("Saved plot:",),
    ),
}


def latest_run_dir(output_root: Path, experiment: str) -> Path | None:
    exp_dir = Path(output_root) / experiment
    runs = sorted(exp_dir.glob("run_*"))
    return runs[-1] if runs else None


def audit_run(run_dir: Path, experiment: str, spec: ExperimentArtifactSpec | None = None) -> RunAuditResult:
    spec = spec or EXPERIMENT_ARTIFACT_SPECS[experiment]
    fig_dir = _figures_dir(run_dir)
    met_dir = _metrics_dir(run_dir)
    art_dir = run_dir / "artifacts"
    log_path = _logs_dir(run_dir) / "run.log"

    present_figures = {path.name for path in fig_dir.glob("*")} if fig_dir.exists() else set()
    present_metrics = {path.name for path in met_dir.glob("*")} if met_dir.exists() else set()
    present_artifacts = {path.name for path in art_dir.glob("*")} if art_dir.exists() else set()
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    missing_figures = tuple(name for name in spec.required_figures if name not in present_figures)
    missing_metrics = tuple(name for name in spec.required_metrics if name not in present_metrics)
    missing_artifacts = tuple(name for name in spec.required_artifacts if name not in present_artifacts)
    missing_completion_markers = tuple(marker for marker in spec.completion_markers if marker not in log_text)
    integrity_issues = spec.validator(run_dir) if spec.validator is not None else ()

    expected_optional_figures: tuple[str, ...] = ()
    if spec.optional_figure_gate is not None and spec.optional_figure_gate(run_dir):
        expected_optional_figures = spec.optional_figures

    return RunAuditResult(
        experiment=experiment,
        run_dir=Path(run_dir),
        missing_figures=missing_figures,
        missing_metrics=missing_metrics,
        missing_artifacts=missing_artifacts,
        missing_completion_markers=missing_completion_markers,
        expected_optional_figures=expected_optional_figures,
        integrity_issues=integrity_issues,
    )


def audit_output_root(output_root: Path, experiments: tuple[str, ...] | None = None) -> list[RunAuditResult]:
    names = experiments or tuple(EXPERIMENT_ARTIFACT_SPECS.keys())
    results: list[RunAuditResult] = []
    for experiment in names:
        run_dir = latest_run_dir(output_root, experiment)
        if run_dir is None:
            continue
        results.append(audit_run(run_dir, experiment))
    return results
