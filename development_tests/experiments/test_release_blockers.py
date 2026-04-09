from pathlib import Path
import re
import sys

import numpy as np
import pytest


def test_check_configs_discovers_profile_configs_and_public_paths():
    from experiments.testing.check_configs import (
        EXPERIMENT_BLOCK_NAMES,
        PUBLIC_EXPERIMENT_BASE_CONFIGS,
        _public_experiment_overrides,
        _resolve_for_validation,
        _discover_root_config_names,
    )
    from hydra import compose, initialize_config_dir
    from gibbsq.core.config import PROFILE_CONFIG_NAMES
    from scripts.execution.experiment_runner import EXPERIMENTS

    profiles = _discover_root_config_names()
    assert profiles == list(PROFILE_CONFIG_NAMES)
    assert tuple(EXPERIMENT_BLOCK_NAMES) == (
        "check_configs",
        "hyperqual",
        "reinforce_check",
        "drift",
        "sweep",
        "stress",
        "policy",
        "bc_train",
        "reinforce_train",
        "stats",
        "generalize",
        "ablation",
        "critical",
    )
    assert _public_experiment_overrides("hyperqual") == ["++active_experiment=hyperqual"]
    assert _public_experiment_overrides("policy") == ["++active_experiment=policy"]
    assert _public_experiment_overrides("stats") == ["++active_experiment=stats"]
    assert _public_experiment_overrides("stress") == ["++active_experiment=stress", "++jax.enabled=True"]
    assert list(PUBLIC_EXPERIMENT_BASE_CONFIGS) == [
        experiment_name for experiment_name in EXPERIMENTS if experiment_name != "check_configs"
    ]
    assert all(PUBLIC_EXPERIMENT_BASE_CONFIGS[name] == "default" for name in PUBLIC_EXPERIMENT_BASE_CONFIGS)

    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        raw_cfg = compose(config_name="final_experiment")
        resolved = _resolve_for_validation(raw_cfg, "ablation", profile_name="final_experiment")
    assert int(resolved.train_epochs) == 15
    assert int(resolved.batch_size) == 16


def test_profile_configs_and_resolved_drift_paths_now_validate():
    from hydra import compose, initialize_config_dir

    from gibbsq.core.config import resolve_experiment_config, hydra_to_config, validate, validate_profile_config

    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for config_name in ("debug", "small", "default", "final_experiment"):
            raw_cfg = compose(config_name=config_name)
            validate_profile_config(raw_cfg)
            resolved = resolve_experiment_config(raw_cfg, "drift", profile_name=config_name)
            validate(hydra_to_config(resolved))


def test_drift_verification_accepts_theorem_backed_policy_paths():
    from experiments.verification.drift_verification import _require_theorem_supported_policy

    assert _require_theorem_supported_policy("softmax") == "raw"
    assert _require_theorem_supported_policy("uas") == "uas"
    with pytest.raises(ValueError, match="certifies only theorem-backed policy paths"):
        _require_theorem_supported_policy("jsq")


def test_public_configs_require_full_stationarity_for_sweep_certification():
    from hydra import compose, initialize_config_dir

    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for config_name in ("debug", "small", "default", "final_experiment"):
            raw_cfg = compose(config_name=config_name)
            assert float(raw_cfg.verification.stationarity_threshold) == pytest.approx(1.0)


def test_reinforce_check_uses_canonical_verification_workload_across_profiles():
    from hydra import compose, initialize_config_dir
    from gibbsq.core.config import load_experiment_config

    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        canonical = None
        for config_name in ("debug", "small", "default", "final_experiment"):
            raw_cfg = compose(config_name=config_name)
            cfg, _ = load_experiment_config(raw_cfg, "reinforce_check", profile_name=config_name)
            fingerprint = (
                cfg.system.num_servers,
                tuple(float(x) for x in cfg.system.service_rates),
                cfg.neural.hidden_size,
                cfg.neural.preprocessing,
                cfg.neural.init_type,
                bool(cfg.jax.enabled),
                float(cfg.verification.gradient_check_sim_time),
                int(cfg.verification.gradient_check_n_samples),
                float(cfg.verification.gradient_check_error_threshold),
            )
            if canonical is None:
                canonical = fingerprint
            assert fingerprint == canonical


def test_gradient_check_fd_uses_trainer_aligned_first_action_objective():
    from experiments.testing.reinforce_gradient_check import _sum_first_action_interval_returns

    class Batch:
        returns = np.array([[10.0, 0.0, 4.0], [0.0, 7.0, 8.0]], dtype=np.float32)
        is_action_mask = np.array([[True, False, True], [False, True, True]])
        valid_mask = np.array([[True, True, False], [True, True, True]])

    total = float(_sum_first_action_interval_returns(Batch()))
    assert total == pytest.approx(17.0)


def test_gradient_check_plot_artifacts_only_use_computed_mask():
    from experiments.testing.reinforce_gradient_check import GradientCheckResult, _build_plot_artifacts

    result = GradientCheckResult(
        reinforce_grad=np.array([1.0, 2.0, 30.0]),
        finite_diff_grad=np.array([1.5, 0.0, -40.0]),
        relative_error=0.0,
        cosine_similarity=1.0,
        bias_estimate=0.0,
        variance_estimate=0.0,
        reinforce_var_vector=np.array([9.0, 9.0, 9.0]),
        reinforce_mean_var_vector=np.array([0.09, 0.09, 0.09]),
        fd_var_vector=np.array([0.16, 0.25, 0.36]),
        computed_mask=np.array([True, False, True]),
        passed=True,
    )

    fd_grads, rf_grads, z_scores = _build_plot_artifacts(result)

    np.testing.assert_allclose(fd_grads, np.array([1.5, -40.0]))
    np.testing.assert_allclose(rf_grads, np.array([1.0, 30.0]))
    np.testing.assert_allclose(
        z_scores,
        np.abs(rf_grads - fd_grads) / np.sqrt(np.array([0.09, 0.09]) + np.array([0.16, 0.36])),
    )


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


def test_policy_comparison_requires_neural_weights(monkeypatch, tmp_path):
    from gibbsq.core.config import ExperimentConfig
    from experiments.evaluation.baselines_comparison import run_corrected_comparison

    monkeypatch.setattr(
        "experiments.evaluation.baselines_comparison.resolve_model_pointer",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing reinforce pointer")),
    )

    cfg = ExperimentConfig()
    cfg.system.num_servers = 2
    cfg.system.service_rates = [1.0, 1.0]
    cfg.system.arrival_rate = 1.0
    cfg.simulation.num_replications = 1
    cfg.simulation.ssa.sim_time = 5.0

    with pytest.raises(FileNotFoundError, match="missing reinforce pointer"):
        run_corrected_comparison(cfg, tmp_path)


def test_policy_grid_rebuilds_neural_eval_policy_for_each_rho(monkeypatch):
    from experiments.evaluation import baselines_comparison as module
    from gibbsq.core.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.system.num_servers = 2
    cfg.system.service_rates = [1.0, 2.0]
    cfg.simulation.num_replications = 1
    cfg.simulation.ssa.sim_time = 2.0
    cfg.generalization.rho_grid_vals = [0.5, 0.9]

    seen_rhos = []

    def fake_build_neural_eval_policy(model, mu, rho=None, mode="deterministic"):
        seen_rhos.append(float(rho))
        return object()

    class DummyResult:
        def __init__(self):
            self.states = np.zeros((2, 2), dtype=np.int64)
            self.times = np.array([0.0, 1.0], dtype=np.float64)
            self.arrival_count = 1
            self.departure_count = 0
            self.final_time = 1.0
            self.num_servers = 2

    monkeypatch.setattr(module, "build_neural_eval_policy", fake_build_neural_eval_policy)
    monkeypatch.setattr(
        module,
        "run_replications",
        lambda **kwargs: [DummyResult()],
    )
    monkeypatch.setattr(
        module,
        "build_policy_by_name",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "time_averaged_queue_lengths",
        lambda result, burn_in_fraction: np.array([1.0, 2.0], dtype=np.float64),
    )
    monkeypatch.setattr(module.pd.DataFrame, "to_csv", lambda self, path, index=False: None)
    monkeypatch.setattr(module, "_plot_platinum_grid", lambda df, output_dir: None)

    module.run_grid_generalization(model=object(), cfg=cfg, run_dir=Path("."))

    assert seen_rhos == [0.5, 0.9]


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


def test_ablation_standard_error_uses_sample_std():
    from experiments.evaluation.n_gibbsq_evals.ablation_ssa import _standard_error

    values = np.array([2.0, 4.0, 8.0], dtype=np.float64)
    expected = np.std(values, ddof=1) / np.sqrt(len(values))
    assert _standard_error(values) == expected


def test_ablation_training_cfg_applies_only_training_overrides():
    from omegaconf import OmegaConf

    from experiments.evaluation.n_gibbsq_evals.ablation_ssa import _build_ablation_training_cfg
    from gibbsq.core.config import ExperimentConfig

    base_cfg = ExperimentConfig()
    base_cfg.train_epochs = 15
    base_cfg.batch_size = 16
    base_cfg.simulation.ssa.sim_time = 1000.0
    base_cfg.neural_training.eval_batches = 5
    base_cfg.neural_training.eval_trajs_per_batch = 10
    base_cfg.neural_training.checkpoint_freq = 25

    raw_cfg = OmegaConf.create(
        {
            "ablation_training": {
                "train_epochs": 8,
                "batch_size": 8,
                "simulation": {"ssa": {"sim_time": 750.0}},
                "neural_training": {
                    "eval_batches": 1,
                    "eval_trajs_per_batch": 3,
                    "checkpoint_freq": 10,
                },
            }
        }
    )

    trainer_cfg = _build_ablation_training_cfg(base_cfg, raw_cfg)

    assert trainer_cfg is not base_cfg
    assert int(base_cfg.train_epochs) == 15
    assert int(base_cfg.batch_size) == 16
    assert float(base_cfg.simulation.ssa.sim_time) == pytest.approx(1000.0)
    assert int(base_cfg.neural_training.eval_batches) == 5
    assert int(base_cfg.neural_training.eval_trajs_per_batch) == 10
    assert int(base_cfg.neural_training.checkpoint_freq) == 25

    assert int(trainer_cfg.train_epochs) == 8
    assert int(trainer_cfg.batch_size) == 8
    assert float(trainer_cfg.simulation.ssa.sim_time) == pytest.approx(750.0)
    assert int(trainer_cfg.neural_training.eval_batches) == 1
    assert int(trainer_cfg.neural_training.eval_trajs_per_batch) == 3
    assert int(trainer_cfg.neural_training.checkpoint_freq) == 10


def test_bc_pretraining_entrypoint_forwards_seed_and_alpha(monkeypatch, tmp_path):
    import experiments.training.pretrain_bc as module
    from gibbsq.core.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.simulation.seed = 321
    cfg.system.alpha = 7.5
    cfg.system.num_servers = 2
    cfg.system.service_rates = [1.0, 1.5]

    seen = {}

    monkeypatch.setattr(module, "load_experiment_config", lambda raw_cfg, experiment_name: (cfg, object()))
    monkeypatch.setattr(module, "get_run_config", lambda cfg, name, raw_cfg: (tmp_path, "run-id"))
    monkeypatch.setattr(module, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "train_robust_bc_policy",
        lambda **kwargs: seen.update(kwargs) or object(),
    )
    monkeypatch.setattr("equinox.tree_serialise_leaves", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "save_model_pointer", lambda **kwargs: None)

    class FakeNet:
        pass

    monkeypatch.setattr(module, "NeuralRouter", lambda **kwargs: FakeNet())

    module.main(object())

    assert seen["seed"] == cfg.simulation.seed
    assert seen["alpha"] == cfg.system.alpha


def test_reinforce_bootstrap_forwards_seed_and_alpha(monkeypatch, tmp_path):
    from experiments.training.train_reinforce import ReinforceTrainer
    from gibbsq.core.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.simulation.seed = 123
    cfg.system.alpha = 4.0
    cfg.system.num_servers = 2
    cfg.system.service_rates = [1.0, 1.5]

    trainer = ReinforceTrainer(cfg, tmp_path, run_logger=None)
    seen = {}

    def fake_train_policy(**kwargs):
        seen["policy"] = kwargs
        return kwargs["policy_net"]

    def fake_train_value(**kwargs):
        seen["value"] = kwargs
        return kwargs["value_net"], None

    monkeypatch.setattr(
        "gibbsq.core.pretraining.train_robust_bc_policy",
        fake_train_policy,
    )
    monkeypatch.setattr(
        "gibbsq.core.pretraining.train_robust_bc_value",
        fake_train_value,
    )

    policy_net = object()
    value_net = object()
    out_policy, out_value = trainer.bootstrap_from_expert(
        policy_net,
        value_net,
        key=None,
        jsq_limit=1.0,
        random_limit=2.0,
        denom=1.0,
    )

    assert out_policy is policy_net
    assert out_value is value_net
    assert seen["policy"]["seed"] == cfg.simulation.seed
    assert seen["policy"]["alpha"] == cfg.system.alpha
    assert seen["value"]["seed"] == cfg.simulation.seed
    assert seen["value"]["alpha"] == cfg.system.alpha


def test_policy_baselines_remove_alpha_opt_label():
    from experiments.evaluation.baselines_comparison import CORRECTED_POLICIES

    labels = [entry["label"] for entry in CORRECTED_POLICIES]
    assert "UAS (alpha=opt)" not in labels


def test_runner_injects_publication_safe_defaults():
    from scripts.execution.experiment_runner import default_hydra_overrides_for_experiment

    assert default_hydra_overrides_for_experiment("drift", []) == ["++active_experiment=drift"]
    assert default_hydra_overrides_for_experiment("hyperqual", []) == ["++active_experiment=hyperqual"]
    assert default_hydra_overrides_for_experiment("sweep", []) == ["++active_experiment=sweep"]
    assert default_hydra_overrides_for_experiment("policy", []) == ["++active_experiment=policy"]
    assert default_hydra_overrides_for_experiment("stats", []) == ["++active_experiment=stats"]
    assert default_hydra_overrides_for_experiment("stress", []) == ["++active_experiment=stress", "++jax.enabled=True"]
    assert default_hydra_overrides_for_experiment("stress", ["++jax.enabled=False"]) == ["++active_experiment=stress"]
    assert default_hydra_overrides_for_experiment("stats", ["++active_experiment=custom"]) == []


def test_final_phase_pipelines_preserve_explicit_config_name(monkeypatch):
    import scripts.execution.final.phase1_pipeline as phase1
    import scripts.execution.final.phase3_pipeline as phase3

    launched: list[tuple[str, list[str]]] = []

    def fake_run_experiment(experiment, hydra_args=None, dry_run=False, progress_mode="auto"):
        launched.append((experiment, list(hydra_args or [])))
        return 0

    class DummyProgress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, *args, **kwargs):
            return None

    monkeypatch.setattr(phase1, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(phase1, "create_progress", lambda **kwargs: DummyProgress())
    monkeypatch.setattr(sys, "argv", ["phase1_pipeline.py", "--config-name", "final_experiment", "--dry-run"])
    assert phase1.main() == 0

    monkeypatch.setattr(phase3, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(phase3, "create_progress", lambda **kwargs: DummyProgress())
    monkeypatch.setattr(sys, "argv", ["phase3_pipeline.py", "--config-name", "final_experiment", "--dry-run"])
    assert phase3.main() == 0

    assert launched
    for _, hydra_args in launched:
        assert hydra_args[:2] == ["--config-name", "final_experiment"]


def test_publication_baselines_follow_active_policy_config():
    from gibbsq.core.config import ExperimentConfig
    from gibbsq.engines.jax_engine import policy_name_to_type

    cfg = ExperimentConfig()
    assert cfg.policy.name == "softmax" or cfg.policy.name == "uas"
    assert policy_name_to_type("softmax") == 3
    assert policy_name_to_type("uas") == 6


def test_stress_heterogeneity_path_uses_active_policy_contract():
    import jax.numpy as jnp

    from experiments.testing.stress_test import _heterogeneity_replication_kwargs
    from gibbsq.core.config import ExperimentConfig, PolicyConfig
    from gibbsq.engines.jax_engine import policy_name_to_type

    cfg = ExperimentConfig()
    cfg.policy = PolicyConfig(name="softmax")
    mu_het = jnp.array([10.0, 0.1, 0.1, 0.1], dtype=jnp.float32)

    kwargs = _heterogeneity_replication_kwargs(
        cfg,
        mu_het=mu_het,
        arrival_rate=0.5,
        sim_time=100.0,
        max_samples=101,
    )

    assert kwargs["policy_type"] == policy_name_to_type("softmax")
    assert kwargs["num_servers"] == len(mu_het)


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


def test_compute_gae_bootstraps_on_predecision_states():
    pytest.importorskip("jax")

    from experiments.training.train_reinforce import compute_gae

    def linear_value_net(Q, mu=None, rho=None):
        return Q[0]

    advantages, returns = compute_gae(
        states=[
            np.array([1.0], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
        ],
        jump_times=[0.0, 1.0, 2.0],
        action_step_indices=[0, 2],
        sim_time=3.0,
        gamma=0.5,
        gae_lambda=0.5,
        value_net=linear_value_net,
        service_rates=np.array([1.0], dtype=np.float32),
        rho=0.7,
        random_limit=10.0,
        denom=10.0,
        decision_states=[
            np.array([10.0], dtype=np.float32),
            np.array([20.0], dtype=np.float32),
        ],
    )

    np.testing.assert_allclose(advantages, np.array([61.03125, 16.5]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(returns, np.array([71.03125, 36.5]), rtol=1e-6, atol=1e-6)
