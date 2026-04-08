import pytest


def test_drift_verification_accepts_theorem_backed_policy_paths():
    from experiments.verification.drift_verification import _require_theorem_supported_policy

    assert _require_theorem_supported_policy("softmax") == "raw"
    assert _require_theorem_supported_policy("uas") == "uas"
    with pytest.raises(ValueError, match="certifies only theorem-backed policy paths"):
        _require_theorem_supported_policy("jsq")


def test_final_phase_pipelines_preserve_explicit_config_name(monkeypatch):
    from scripts.execution.final import phase1_pipeline, phase3_pipeline

    captured = []

    def fake_run_experiment(experiment, current_args, *, dry_run, progress_mode):
        captured.append(
            {
                "experiment": experiment,
                "current_args": tuple(current_args),
                "dry_run": dry_run,
                "progress_mode": progress_mode,
            }
        )
        return 0

    monkeypatch.setattr(phase1_pipeline, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(phase3_pipeline, "run_experiment", fake_run_experiment)

    monkeypatch.setattr(
        "sys.argv",
        ["phase1_pipeline.py", "--config-name", "final_experiment", "--dry-run"],
    )
    assert phase1_pipeline.main() == 0
    monkeypatch.setattr(
        "sys.argv",
        ["phase3_pipeline.py", "--config-name", "final_experiment", "--dry-run"],
    )
    assert phase3_pipeline.main() == 0

    assert len(captured) == len(phase1_pipeline.PHASE_1_EXPERIMENTS) + len(phase3_pipeline.PHASE_3_EXPERIMENTS)
    for call in captured:
        assert "--config-name" in call["current_args"]
        idx = call["current_args"].index("--config-name")
        assert call["current_args"][idx + 1] == "final_experiment"
        assert call["dry_run"] is True
        assert call["progress_mode"] == "auto"
