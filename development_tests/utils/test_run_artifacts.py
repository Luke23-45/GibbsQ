from types import SimpleNamespace
from pathlib import Path

from omegaconf import OmegaConf

from gibbsq.utils.discovery import load_metrics
from gibbsq.utils.exporter import append_metrics_jsonl
from gibbsq.utils.artifact_audit import audit_output_root, audit_run
from gibbsq.utils.logging import get_run_config
from gibbsq.utils.run_artifacts import (
    artifacts_dir,
    config_path,
    figure_path,
    figures_dir,
    logs_dir,
    metadata_dir,
    metadata_path,
    metrics_dir,
    metrics_path,
)


def _cfg(tmp_path: Path):
    return SimpleNamespace(
        jax=SimpleNamespace(enabled=False, platform="cpu", precision="float32", fallback_to_cpu=True),
        output_dir=str(tmp_path / "outputs" / "small"),
        wandb=SimpleNamespace(run_name=None),
    )


def test_get_run_config_creates_standard_capsule(tmp_path):
    cfg = _cfg(tmp_path)
    raw_cfg = OmegaConf.create({"output_dir": cfg.output_dir, "wandb": {"run_name": None}})

    run_dir, run_id = get_run_config(cfg, "stress", raw_cfg)

    assert run_id.startswith("run_")
    assert logs_dir(run_dir).is_dir()
    assert figures_dir(run_dir).is_dir()
    assert metrics_dir(run_dir).is_dir()
    assert artifacts_dir(run_dir).is_dir()
    assert metadata_dir(run_dir).is_dir()
    assert config_path(run_dir).exists()


def test_run_artifact_helpers_build_expected_paths(tmp_path):
    run_dir = tmp_path / "outputs" / "small" / "stress" / "run_20260403_000000"

    assert figure_path(run_dir, "stress_dashboard") == run_dir / "figures" / "stress_dashboard"
    assert metrics_path(run_dir) == run_dir / "metrics" / "metrics.jsonl"
    assert metadata_path(run_dir, "study_config.json") == run_dir / "metadata" / "study_config.json"
    assert config_path(run_dir) == run_dir / "metadata" / "config.yaml"


def test_load_metrics_reads_new_and_legacy_locations(tmp_path):
    new_run = tmp_path / "new_run"
    append_metrics_jsonl({"value": 1}, metrics_path(new_run))
    new_df = load_metrics(new_run)
    assert list(new_df["value"]) == [1]

    legacy_run = tmp_path / "legacy_run"
    append_metrics_jsonl({"value": 2}, legacy_run / "metrics.jsonl")
    legacy_df = load_metrics(legacy_run)
    assert list(legacy_df["value"]) == [2]


def test_audit_run_detects_missing_required_critical_curve(tmp_path):
    run_dir = tmp_path / "outputs" / "debug" / "critical" / "run_20260403_000000"
    logs_dir(run_dir).mkdir(parents=True, exist_ok=True)
    figures_dir(run_dir).mkdir(parents=True, exist_ok=True)
    metrics_dir(run_dir).mkdir(parents=True, exist_ok=True)
    metadata_dir(run_dir).mkdir(parents=True, exist_ok=True)

    metrics_path(run_dir).write_text('{"rho": 0.95}\n', encoding="utf-8")
    config_path(run_dir).write_text("output_dir: outputs/debug\n", encoding="utf-8")
    (logs_dir(run_dir) / "run.log").write_text(
        "Evaluating Boundary rho=0.950\n",
        encoding="utf-8",
    )

    result = audit_run(run_dir, "critical")

    assert "critical_load_curve.png" in result.missing_figures
    assert "critical_load_curve.pdf" in result.missing_figures
    assert "Critical load test complete. Curve saved" in result.missing_completion_markers
    assert not result.ok


def test_audit_output_root_handles_optional_policy_grid(tmp_path):
    run_dir = tmp_path / "outputs" / "debug" / "policy" / "run_20260403_000000"
    logs_dir(run_dir).mkdir(parents=True, exist_ok=True)
    figures_dir(run_dir).mkdir(parents=True, exist_ok=True)
    metrics_dir(run_dir).mkdir(parents=True, exist_ok=True)
    metadata_dir(run_dir).mkdir(parents=True, exist_ok=True)

    for name in ("corrected_policy_comparison.png", "corrected_policy_comparison.pdf"):
        (figures_dir(run_dir) / name).write_text("ok", encoding="utf-8")
    (metrics_dir(run_dir) / "corrected_comparison_metrics.jsonl").write_text(
        '{"policy":"A","tier":2}\n{"policy":"N-GibbsQ (Platinum)","tier":5}\n',
        encoding="utf-8",
    )
    (logs_dir(run_dir) / "run.log").write_text("Comparison plot saved\n", encoding="utf-8")
    config_path(run_dir).write_text("grid: true\n", encoding="utf-8")

    [result] = audit_output_root(tmp_path / "outputs" / "debug", experiments=("policy",))

    assert result.experiment == "policy"
    assert result.expected_optional_figures == (
        "platinum_grid_analysis.png",
        "platinum_grid_analysis.pdf",
    )
    assert result.ok


def test_audit_run_flags_policy_metrics_without_tier5(tmp_path):
    run_dir = tmp_path / "outputs" / "debug" / "policy" / "run_20260403_000000"
    logs_dir(run_dir).mkdir(parents=True, exist_ok=True)
    figures_dir(run_dir).mkdir(parents=True, exist_ok=True)
    metrics_dir(run_dir).mkdir(parents=True, exist_ok=True)
    metadata_dir(run_dir).mkdir(parents=True, exist_ok=True)

    for name in ("corrected_policy_comparison.png", "corrected_policy_comparison.pdf"):
        (figures_dir(run_dir) / name).write_text("ok", encoding="utf-8")
    (metrics_dir(run_dir) / "corrected_comparison_metrics.jsonl").write_text(
        '{"policy":"JSSQ (Min Sojourn)","tier":2}\n',
        encoding="utf-8",
    )
    (logs_dir(run_dir) / "run.log").write_text("Comparison plot saved\n", encoding="utf-8")
    config_path(run_dir).write_text("grid: false\n", encoding="utf-8")

    result = audit_run(run_dir, "policy")

    assert "policy metrics missing tier=5 row" in result.integrity_issues
    assert "policy metrics missing N-GibbsQ (Platinum) row" in result.integrity_issues
    assert not result.ok


def test_audit_run_flags_reinforce_stage_profile_without_final_eval(tmp_path):
    run_dir = tmp_path / "outputs" / "debug" / "reinforce_train" / "run_20260403_000000"
    logs_dir(run_dir).mkdir(parents=True, exist_ok=True)
    figures_dir(run_dir).mkdir(parents=True, exist_ok=True)
    metrics_dir(run_dir).mkdir(parents=True, exist_ok=True)
    metadata_dir(run_dir).mkdir(parents=True, exist_ok=True)

    for name in ("reinforce_training_curve.png", "reinforce_training_curve.pdf"):
        (figures_dir(run_dir) / name).write_text("ok", encoding="utf-8")
    (metrics_dir(run_dir) / "reinforce_metrics.jsonl").write_text('{"epoch":0}\n', encoding="utf-8")
    (metrics_dir(run_dir) / "reinforce_stage_profile.json").write_text('{"epochs":[]}\n', encoding="utf-8")
    (logs_dir(run_dir) / "run.log").write_text("Stage profile written\nTraining Complete!\n", encoding="utf-8")
    config_path(run_dir).write_text("output_dir: outputs/debug\n", encoding="utf-8")

    result = audit_run(run_dir, "reinforce_train")

    assert "reinforce stage profile missing final_eval summary" in result.integrity_issues
    assert not result.ok
