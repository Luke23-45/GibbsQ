from pathlib import Path
import re
import sys

import numpy as np
import pytest


def test_check_configs_discovers_experiment_profiles():
    from experiments.testing.check_configs import (
        PUBLIC_EXPERIMENT_BASE_CONFIGS,
        _public_experiment_overrides,
        _base_config_for_experiment,
        _discover_experiment_profiles,
    )
    from scripts.execution.experiment_runner import EXPERIMENTS

    profiles = _discover_experiment_profiles()
    assert profiles == [
        "neural_eval",
        "policy_comparison",
        "stability_sweep",
        "stats_bench",
    ]
    assert _base_config_for_experiment("stability_sweep") == "default"
    assert _base_config_for_experiment("stability_sweep_small") == "small"
    assert _base_config_for_experiment("stability_sweep_large") == "large"
    assert _public_experiment_overrides("stress") == ["++jax.enabled=True"]
    assert _public_experiment_overrides("policy") == ["+experiment=policy_comparison"]
    assert _public_experiment_overrides("stats") == ["+experiment=stats_bench"]
    assert list(PUBLIC_EXPERIMENT_BASE_CONFIGS) == [
        experiment_name for experiment_name in EXPERIMENTS if experiment_name != "check_configs"
    ]
    assert PUBLIC_EXPERIMENT_BASE_CONFIGS["drift"] == "drift"
    assert all(
        PUBLIC_EXPERIMENT_BASE_CONFIGS[name] == "default"
        for name in PUBLIC_EXPERIMENT_BASE_CONFIGS
        if name != "drift"
    )


def test_drift_verification_rejects_non_softmax_theorem_claims():
    from experiments.verification.drift_verification import _require_theorem_supported_policy

    assert _require_theorem_supported_policy("softmax") == "raw"
    with pytest.raises(ValueError, match="certifies only the raw softmax theorem path"):
        _require_theorem_supported_policy("uas")


def test_gradient_check_fd_uses_same_action_interval_objective():
    from experiments.testing.reinforce_gradient_check import _sum_masked_action_interval_returns

    class Batch:
        returns = np.array([[10.0, 0.0, 4.0], [0.0, 7.0, 8.0]], dtype=np.float32)
        is_action_mask = np.array([[True, False, True], [False, True, True]])
        valid_mask = np.array([[True, True, False], [True, True, True]])

    total = float(_sum_masked_action_interval_returns(Batch()))
    assert total == pytest.approx(25.0)


def test_stats_benchmark_load_helper_fails_closed(monkeypatch):
    from experiments.evaluation.n_gibbsq_evals.stats_bench import _load_trained_model_or_fail

    class DummyCfg:
        neural = object()

    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("missing pointer")

    monkeypatch.setattr(
        "experiments.evaluation.n_gibbsq_evals.stats_bench.resolve_model_pointer",
        raise_missing,
    )

    with pytest.raises(FileNotFoundError, match="missing pointer"):
        _load_trained_model_or_fail(
            DummyCfg(),
            num_servers=2,
            service_rates=np.array([1.0, 1.0], dtype=np.float32),
            load_key=None,
            project_root=Path("temp"),
            output_root=Path("temp"),
        )


def test_ablation_uses_deterministic_eval_policy():
    from experiments.evaluation.n_gibbsq_evals.ablation_ssa import _build_ablation_eval_policy
    from gibbsq.utils.model_io import DeterministicNeuralPolicy

    class MockNet:
        def get_numpy_params(self):
            return [(np.eye(2), np.zeros(2))]

        def numpy_forward(self, x, params, config, **kwargs):
            return x

        config = type("Config", (), {"preprocessing": "none", "use_rho": False})()

    policy = _build_ablation_eval_policy(MockNet(), np.array([1.0, 1.0]), rho=0.8)
    assert isinstance(policy, DeterministicNeuralPolicy)


def test_ablation_variants_are_canonical_and_distinct():
    from experiments.evaluation.n_gibbsq_evals.ablation_ssa import _variant_cfg
    from gibbsq.core.config import ExperimentConfig, NeuralConfig

    base_cfg = ExperimentConfig()
    base_cfg.neural = NeuralConfig(preprocessing="none", init_type="xavier_uniform")

    full_cfg = _variant_cfg(base_cfg, None, None)
    no_log_cfg = _variant_cfg(base_cfg, "none", None)
    no_init_cfg = _variant_cfg(base_cfg, None, "standard")

    assert full_cfg.neural.preprocessing == "log1p"
    assert full_cfg.neural.init_type == "zero_final"
    assert no_log_cfg.neural.preprocessing == "none"
    assert no_log_cfg.neural.init_type == "zero_final"
    assert no_init_cfg.neural.preprocessing == "log1p"
    assert no_init_cfg.neural.init_type == "standard"


def test_policy_baselines_remove_alpha_opt_label():
    from experiments.evaluation.baselines_comparison import CORRECTED_POLICIES

    labels = [entry["label"] for entry in CORRECTED_POLICIES]
    assert "UAS (alpha=opt)" not in labels


def test_runner_injects_publication_safe_defaults():
    from scripts.execution.experiment_runner import default_hydra_overrides_for_experiment

    assert default_hydra_overrides_for_experiment("drift", []) == []
    assert default_hydra_overrides_for_experiment("sweep", []) == ["+experiment=stability_sweep"]
    assert default_hydra_overrides_for_experiment("policy", []) == ["+experiment=policy_comparison"]
    assert default_hydra_overrides_for_experiment("stats", []) == ["+experiment=stats_bench"]
    assert default_hydra_overrides_for_experiment("stress", []) == ["++jax.enabled=True"]
    assert default_hydra_overrides_for_experiment("stress", ["++jax.enabled=False"]) == []
    assert default_hydra_overrides_for_experiment("stats", ["+experiment=custom"]) == []


def test_publication_baselines_follow_active_policy_config():
    from gibbsq.core.config import ExperimentConfig
    from gibbsq.engines.jax_engine import policy_name_to_type

    cfg = ExperimentConfig()
    assert cfg.policy.name == "softmax" or cfg.policy.name == "uas"
    assert policy_name_to_type("softmax") == 3
    assert policy_name_to_type("uas") == 6


def test_jax_policy_type_helper_rejects_unknown_names():
    from gibbsq.engines.jax_engine import policy_name_to_type

    with pytest.raises(ValueError, match="Unsupported JAX policy name"):
        policy_name_to_type("does_not_exist")


def test_reproduction_pipeline_stops_before_dependent_eval_phase(monkeypatch, capsys):
    import scripts.execution.reproduction_pipeline as pipeline

    launched = []

    def fake_run_experiment(experiment, hydra_args=None, dry_run=False, progress_mode="auto"):
        launched.append(experiment)
        return 1 if experiment == "reinforce_train" else 0

    class DummyProgress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_description(self, *args, **kwargs):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def update(self, *args, **kwargs):
            return None

    monkeypatch.setattr(pipeline, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(pipeline, "create_progress", lambda **kwargs: DummyProgress())
    monkeypatch.setattr(sys, "argv", ["reproduction_pipeline.py"])

    exit_code = pipeline.main()

    assert exit_code == 1
    assert "reinforce_train" in launched
    assert "policy" not in launched
    assert "stats" not in launched
    captured = capsys.readouterr()
    assert "Stopping pipeline after the first failure" in captured.out


def test_public_runner_aliases_match_run_capsule_names():
    from scripts.execution.experiment_runner import EXPERIMENTS

    expected_modules = {
        module_path: alias
        for alias, module_path in EXPERIMENTS.items()
        if alias != "check_configs"
    }
    expected_modules["experiments.evaluation.n_gibbsq_evals.ablation_ssa"] = "ablation"
    expected_modules["experiments.evaluation.n_gibbsq_evals.critical_load"] = "critical"

    for module_path, alias in expected_modules.items():
        file_path = Path(module_path.replace(".", "/") + ".py")
        source = file_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'get_run_config\(cfg,\s*"([^"]+)"', source)
        assert match is not None, f"Missing get_run_config call in {file_path}"
        assert match.group(1) == alias, f"{file_path} should write outputs under alias '{alias}'"


def test_reproduction_pipeline_progress_labels_are_consistent():
    from scripts.execution.reproduction_pipeline import _format_pipeline_step

    assert _format_pipeline_step("Running Configuration Sanity Checks...", None, 11) == "[Pre-Flight] Running Configuration Sanity Checks..."
    assert _format_pipeline_step("Running Drift Verification...", 2, 11) == "[2/11] Running Drift Verification..."


def test_validation_suite_progress_labels_are_consistent():
    from scripts.verification.validation_suite import _format_validation_step

    assert _format_validation_step(1, 6, "Running pre-flight configuration check...") == "[1/6] Running pre-flight configuration check..."
    assert _format_validation_step(5, 5, "Running platinum parity benchmark...") == "[5/5] Running platinum parity benchmark..."


def test_public_progress_labels_drop_stale_track_wording():
    reproduction_source = Path("scripts/execution/reproduction_pipeline.py").read_text(encoding="utf-8", errors="replace")
    validation_source = Path("scripts/verification/validation_suite.py").read_text(encoding="utf-8", errors="replace")

    assert "Track 5" not in reproduction_source
    assert "Track 5" not in validation_source
