import pytest
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf


def test_drift_verification_accepts_theorem_backed_policy_paths():
    from experiments.verification.drift_verification import _require_theorem_supported_policy

    assert _require_theorem_supported_policy("softmax") == "raw"
    assert _require_theorem_supported_policy("uas") == "uas"
    with pytest.raises(ValueError, match="certifies only theorem-backed policy paths"):
        _require_theorem_supported_policy("jsq")
    with pytest.raises(ValueError, match="certifies only theorem-backed policy paths"):
        _require_theorem_supported_policy("calibrated_uas")


def test_stability_sweep_remains_theorem_only():
    allowed = ["softmax", "uas"]

    assert "calibrated_uas" not in allowed


def test_publication_neural_evaluation_runners_pin_calibrated_uas_baseline():
    from experiments.evaluation.n_gibbsq_evals.critical_load import _publication_baseline_spec as critical_spec
    from experiments.evaluation.n_gibbsq_evals.gen_sweep import _publication_baseline_spec as generalize_spec
    from experiments.evaluation.n_gibbsq_evals.stats_bench import _publication_baseline_spec as stats_spec

    assert stats_spec()[0] == "calibrated_uas"
    assert generalize_spec()[0] == "calibrated_uas"
    assert critical_spec()[0] == "calibrated_uas"


@pytest.mark.parametrize(
    "config_name",
    ["debug", "small", "default", "final_experiment"],
)
def test_publication_comparison_configs_disable_jax_for_mixed_backend_runners(config_name):
    cfg = OmegaConf.load(Path("configs") / f"{config_name}.yaml")

    assert cfg.experiments.stats.jax.enabled is False
    assert cfg.experiments.generalize.jax.enabled is False
    assert cfg.experiments.critical.jax.enabled is False


def test_engine_parity_experiment_is_registered_and_policy_set_is_publication_safe():
    from scripts.execution.experiment_runner import EXPERIMENTS
    from gibbsq.core.config import EXPERIMENT_BLOCK_NAMES
    from experiments.verification.engine_parity import _policy_label, _policy_contract_alpha

    assert "engine_parity" in EXPERIMENTS
    assert "engine_parity" in EXPERIMENT_BLOCK_NAMES
    assert _policy_label("calibrated_uas") == "Calibrated UAS"
    assert _policy_contract_alpha("uas", 1.0) == pytest.approx(10.0)
    assert _policy_contract_alpha("calibrated_uas", 1.0) == pytest.approx(20.0)
    assert _policy_contract_alpha("jsq", 1.0) == pytest.approx(1.0)


def test_hyperqual_summarizes_clean_policy_labels():
    from gibbsq.studies.hyperparameter_qualification import summarize_policy_results

    summary = summarize_policy_results(
        {
            "N-GibbsQ (Proposed)": {"mean_q_total": 10.1, "parity": "SILVER"},
            "JSSQ (Min Sojourn)": {"mean_q_total": 11.0},
            "UAS": {"mean_q_total": 11.5},
            "Calibrated UAS": {"mean_q_total": 10.0},
        }
    )

    assert summary["parity"] == "SILVER"
    assert summary["neural_mean_q_total"] == pytest.approx(10.1)
    assert summary["jssq_mean_q_total"] == pytest.approx(11.0)
    assert summary["uas_mean_q_total"] == pytest.approx(11.5)
    assert summary["calibrated_uas_mean_q_total"] == pytest.approx(10.0)


def test_critical_regeneration_prefers_calibrated_uas_metric(monkeypatch):
    from gibbsq.analysis import critical_regeneration

    workspace_tmp = Path("tests") / "_tmp_publication_release"
    if workspace_tmp.exists():
        import shutil
        shutil.rmtree(workspace_tmp)
    run_dir = workspace_tmp / "critical_run"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.jsonl").write_text(
        '{"rho": 0.9, "neural_eq": 9.8, "calibrated_uas_eq": 10.2, "gibbs_eq": 11.1}\n',
        encoding="utf-8",
    )

    captured = {}

    def fake_plot_critical_load(*, rho_values, neural_eq, gibbs_eq, save_path, theme, formats):
        captured["rho_values"] = rho_values
        captured["neural_eq"] = neural_eq
        captured["gibbs_eq"] = gibbs_eq

    monkeypatch.setattr(critical_regeneration, "plot_critical_load", fake_plot_critical_load)

    critical_regeneration.regenerate_critical_figure(run_dir)

    assert list(captured["rho_values"]) == [0.9]
    assert list(captured["neural_eq"]) == [9.8]
    assert list(captured["gibbs_eq"]) == [10.2]

    import shutil
    shutil.rmtree(workspace_tmp)


def test_critical_regeneration_requires_v3_calibrated_uas_metric():
    from gibbsq.analysis.critical_regeneration import regenerate_critical_figure

    workspace_tmp = Path("tests") / "_tmp_publication_release_missing_v3"
    if workspace_tmp.exists():
        import shutil
        shutil.rmtree(workspace_tmp)
    run_dir = workspace_tmp / "critical_run"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.jsonl").write_text(
        '{"rho": 0.9, "neural_eq": 9.8, "baseline_eq": 10.2, "gibbs_eq": 11.1}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires v3 metrics"):
        regenerate_critical_figure(run_dir)

    import shutil
    shutil.rmtree(workspace_tmp)


def test_critical_load_keeps_plot_baseline_series_numeric(monkeypatch):
    from experiments.evaluation.n_gibbsq_evals import critical_load

    cfg = SimpleNamespace(
        system=SimpleNamespace(num_servers=2, service_rates=[1.0, 1.5], alpha=20.0),
        simulation=SimpleNamespace(
            ssa=SimpleNamespace(sim_time=10.0, sample_interval=1.0),
            num_replications=2,
            seed=7,
            burn_in_fraction=0.0,
        ),
        generalization=SimpleNamespace(rho_boundary_vals=[0.9], rho_boundary_threshold=0.95),
        neural=SimpleNamespace(),
    )

    captured = {}

    class DummySimResult:
        pass

    monkeypatch.setattr(critical_load, "resolve_model_pointer", lambda *args, **kwargs: Path("dummy.eqx"))
    monkeypatch.setattr(critical_load, "NeuralRouter", lambda *args, **kwargs: object())
    monkeypatch.setattr(critical_load.eqx, "tree_deserialise_leaves", lambda *args, **kwargs: object())
    monkeypatch.setattr("gibbsq.utils.model_io.validate_neural_model_shape", lambda *args, **kwargs: None)
    monkeypatch.setattr(critical_load, "build_policy_by_name", lambda *args, **kwargs: object())
    monkeypatch.setattr(critical_load, "build_neural_eval_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(critical_load, "compute_poisson_max_steps", lambda *args, **kwargs: 100)
    monkeypatch.setattr(critical_load, "critical_load_sim_time", lambda *args, **kwargs: 5.0)
    monkeypatch.setattr(
        critical_load,
        "run_replications",
        lambda **kwargs: [DummySimResult(), DummySimResult()],
    )
    monkeypatch.setattr(critical_load, "simulate", lambda **kwargs: DummySimResult())
    monkeypatch.setattr(critical_load, "time_averaged_queue_lengths", lambda *args, **kwargs: critical_load.np.array([3.0, 4.0]))
    monkeypatch.setattr(
        critical_load.CriticalLoadTest,
        "_assert_curve_artifacts",
        lambda self: None,
    )

    def fake_plot_progress(self, rho_vals, neural_r, baseline_r):
        captured["rho_vals"] = list(rho_vals)
        captured["neural_r"] = list(neural_r)
        captured["baseline_r"] = list(baseline_r)

    monkeypatch.setattr(critical_load.CriticalLoadTest, "_plot_progress", fake_plot_progress)

    test = critical_load.CriticalLoadTest(cfg, Path("tests/_tmp_critical_numeric"), None)
    result = test.execute(critical_load.jax.random.PRNGKey(cfg.simulation.seed))

    assert captured["rho_vals"] == [0.9]
    assert captured["neural_r"] == [7.0]
    assert captured["baseline_r"] == [7.0]
    assert all(isinstance(value, float) for value in captured["baseline_r"])
    assert result["calibrated_uas_eq"] == [7.0]


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
