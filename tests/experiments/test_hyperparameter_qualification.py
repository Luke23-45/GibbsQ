from pathlib import Path

from omegaconf import OmegaConf

from gibbsq.core.pretraining import extract_bc_data_config
from gibbsq.studies import hyperparameter_qualification as hq


def test_extract_bc_data_config_reads_raw_mapping():
    raw = OmegaConf.create(
        {
            "bc_data": {
                "rhos": [0.4, 0.7],
                "mu_scales": [1.0, 2.0],
                "expert_sim_time": 750.0,
                "augmentation_noise_min": -2,
                "augmentation_noise_max": 2,
            }
        }
    )

    cfg = extract_bc_data_config(raw)

    assert cfg["rhos"] == [0.4, 0.7]
    assert cfg["mu_scales"] == [1.0, 2.0]
    assert cfg["expert_sim_time"] == 750.0
    assert cfg["augmentation_noise_min"] == -2
    assert cfg["augmentation_noise_max"] == 2


def test_rank_candidates_prefers_robust_candidate():
    rows = [
        {
            "candidate_id": "hero",
            "passes_all_gates": False,
            "worst_seed_performance": 95.0,
            "worst_condition_score": 0.2,
            "median_parity_score": 3.0,
            "median_generalization_score": 1.3,
            "mean_score": 9.0,
        },
        {
            "candidate_id": "robust",
            "passes_all_gates": True,
            "worst_seed_performance": 80.0,
            "worst_condition_score": 1.0,
            "median_parity_score": 2.0,
            "median_generalization_score": 1.1,
            "mean_score": 6.5,
        },
    ]

    ranked = hq.rank_candidates(rows)

    assert ranked[0]["candidate_id"] == "robust"
    assert ranked[0]["rank"] == 1


def test_run_stage_writes_artifacts(monkeypatch, tmp_path):
    study_cfg = hq.StudyConfig(stage="A", mode="full", candidate_count=2, promote_top_k=1, seed_values=[11, 12])
    preset = hq.STAGE_PRESETS["A"]

    def fake_execute_seed_trial(*, raw_candidate, profile_name, suite, mode):
        seed = int(OmegaConf.select(raw_candidate, "simulation.seed"))
        return {
            "seed": seed,
            "training": {
                "final_performance_index_ema": 80.0 + seed,
                "has_nans": False,
                "max_policy_grad_norm": 10.0,
                "max_value_grad_norm": 12.0,
                "monotone_fraction": 0.75,
                "improved": True,
            },
            "policy": {"parity_numeric": 2},
            "generalization": {
                "min_improvement_ratio": 1.05,
                "mean_improvement_ratio": 1.10,
            },
        }

    monkeypatch.setattr(hq, "execute_seed_trial", fake_execute_seed_trial)

    result = hq.run_stage(stage="A", study_cfg=study_cfg, preset=preset, hyperqual_run_dir=tmp_path)

    stage_dir = Path(result["output_dir"])
    assert stage_dir.exists()
    assert (stage_dir / "taxonomy.json").exists()
    assert (stage_dir / "bc_leaderboard.json").exists()
    assert (stage_dir / "final_recommendation.md").exists()
    assert result["leaderboard"][0]["rank"] == 1


def test_eval_only_scorecard_does_not_require_training_metrics():
    study_cfg = hq.StudyConfig(stage="A", mode="eval_only")
    trial = {
        "seed": 17,
        "policy": {"parity_numeric": 2},
        "generalization": {
            "min_improvement_ratio": 1.02,
            "mean_improvement_ratio": 1.08,
        },
    }

    scorecard = hq.compute_trial_scorecard(
        trial,
        study_cfg=study_cfg,
        suite=["policy", "generalize"],
    )

    assert scorecard["training_pass"] is True
    assert scorecard["policy_pass"] is True
    assert scorecard["generalization_pass"] is True
    assert scorecard["all_gates_pass"] is True
    assert scorecard["worst_seed_performance"] > 0.0


def test_run_stage_full_certification_and_wandb_flags(monkeypatch, tmp_path):
    study_cfg = hq.StudyConfig(
        stage="A",
        mode="eval_only",
        candidate_count=1,
        promote_top_k=1,
        seed_values=[7],
        full_certification=True,
        allow_wandb=True,
    )
    preset = hq.STAGE_PRESETS["A"]
    seen = {}

    def fake_execute_seed_trial(*, raw_candidate, profile_name, suite, mode):
        seen["suite"] = list(suite)
        seen["wandb_enabled"] = bool(OmegaConf.select(raw_candidate, "wandb.enabled"))
        seen["wandb_run_name"] = OmegaConf.select(raw_candidate, "wandb.run_name")
        return {
            "seed": int(OmegaConf.select(raw_candidate, "simulation.seed")),
            "policy": {"parity_numeric": 2},
            "generalization": {
                "min_improvement_ratio": 1.05,
                "mean_improvement_ratio": 1.10,
            },
            "critical": {
                "max_neural_to_gibbs_ratio": 1.0,
                "mean_neural_to_gibbs_ratio": 1.0,
            },
            "stress": {
                "min_stationarity_rate": 1.0,
                "mean_stationarity_rate": 1.0,
                "heterogeneity_gini": 0.1,
            },
        }

    monkeypatch.setattr(hq, "execute_seed_trial", fake_execute_seed_trial)

    result = hq.run_stage(stage="A", study_cfg=study_cfg, preset=preset, hyperqual_run_dir=tmp_path)

    assert seen["suite"] == ["policy", "generalize", "critical", "stress"]
    assert seen["wandb_enabled"] is True
    assert seen["wandb_run_name"].startswith("a-")
    assert seen["wandb_run_name"].endswith("-s7")
    assert result["leaderboard"][0]["passes_all_gates"] is True
