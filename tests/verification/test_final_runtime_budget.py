from __future__ import annotations

import json
from pathlib import Path

from scripts.verification import final_runtime_budget
from scripts.verification.runtime_budgeting import (
    ALL_GROUPED_EXPERIMENTS,
    GROUP_A_EXPERIMENTS,
    GROUP_B_EXPERIMENTS,
    GROUP_C_EXPERIMENTS,
    default_calibration_matrix,
    latest_complete_sections_by_config,
)


def test_group_partition_covers_all_expected_experiments():
    assert len(ALL_GROUPED_EXPERIMENTS) == len(set(ALL_GROUPED_EXPERIMENTS))
    assert GROUP_A_EXPERIMENTS + GROUP_B_EXPERIMENTS + GROUP_C_EXPERIMENTS == ALL_GROUPED_EXPERIMENTS
    assert len(ALL_GROUPED_EXPERIMENTS) == 12


def test_log_parser_picks_latest_complete_section(tmp_path: Path):
    log_path = tmp_path / "mixed_logs.md"
    log_path.write_text(
        "\n".join(
            [
                "PS C:\\repo> python scripts/execution/reproduction_pipeline.py --config-name debug",
                "[Experiment 'check_configs' Finished]",
                "  Elapsed Duration: 10.000s",
                "  Pipeline Status: interrupted by user",
                "PS C:\\repo> python scripts/execution/reproduction_pipeline.py --config-name debug",
                "[Experiment 'check_configs' Finished]",
                "  Elapsed Duration: 12.000s",
                "[Experiment 'reinforce_train' Finished]",
                "  Elapsed Duration: 80.000s",
                "  Pipeline Status: completed",
            ]
        ),
        encoding="utf-8",
    )

    sections = latest_complete_sections_by_config([log_path])
    assert set(sections) == {"debug"}
    debug = sections["debug"]
    assert debug.completed is True
    assert debug.timings["check_configs"] == 12.0
    assert debug.timings["reinforce_train"] == 80.0


def test_default_reinforce_calibration_matrix_has_expected_knobs():
    rows = default_calibration_matrix("small", "reinforce_train")
    keys = {tuple(sorted(row.keys())) for row in rows}
    assert ("batch_size",) in keys
    assert ("eval_batches", "eval_trajs_per_batch") in keys
    assert ("sim_time",) in keys
    assert ("train_epochs",) in keys


def test_build_runtime_plan_uses_log_anchors(tmp_path: Path, monkeypatch):
    debug_log = tmp_path / "debug_logs.md"
    small_log = tmp_path / "small_logs.md"
    debug_log.write_text(
        "\n".join(
            [
                "PS C:\\repo> python scripts/execution/reproduction_pipeline.py --config-name debug",
                "[Experiment 'reinforce_train' Finished]",
                "  Elapsed Duration: 85.000s",
                "[Experiment 'generalize' Finished]",
                "  Elapsed Duration: 70.000s",
                "[Experiment 'critical' Finished]",
                "  Elapsed Duration: 100.000s",
                "[Experiment 'stress' Finished]",
                "  Elapsed Duration: 18.000s",
                "  Pipeline Status: completed",
            ]
        ),
        encoding="utf-8",
    )
    small_log.write_text(
        "\n".join(
            [
                "PS C:\\repo> python scripts/execution/reproduction_pipeline.py --config-name small",
                "[Experiment 'reinforce_train' Finished]",
                "  Elapsed Duration: 88.000s",
                "[Experiment 'generalize' Finished]",
                "  Elapsed Duration: 340.000s",
                "[Experiment 'critical' Finished]",
                "  Elapsed Duration: 400.000s",
                "[Experiment 'ablation' Finished]",
                "  Elapsed Duration: 280.000s",
                "[Experiment 'stress' Finished]",
                "  Elapsed Duration: 83.000s",
                "  Pipeline Status: completed",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(final_runtime_budget, "LOG_PATHS", (debug_log, small_log))
    current, candidates, summary = final_runtime_budget.build_runtime_plan(
        calibration_dir=tmp_path / "calibration",
        standalone_budget_minutes=240.0,
    )
    assert "reinforce_train" in current["experiments"]
    assert current["experiments"]["reinforce_train"]["predicted_seconds"] >= 0.0
    assert current["experiments"]["ablation"]["predicted_seconds"] >= 0.0
    assert "Budget Candidates" in summary
    assert isinstance(candidates["generalize"], list)


def test_final_runtime_budget_main_writes_outputs(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "planner_outputs"
    debug_log = tmp_path / "debug_logs.md"
    small_log = tmp_path / "small_logs.md"
    for path in (debug_log, small_log):
        path.write_text(
            "\n".join(
                [
                    "PS C:\\repo> python scripts/execution/reproduction_pipeline.py --config-name small",
                    "[Experiment 'reinforce_train' Finished]",
                    "  Elapsed Duration: 88.000s",
                    "[Experiment 'generalize' Finished]",
                    "  Elapsed Duration: 340.000s",
                    "[Experiment 'critical' Finished]",
                    "  Elapsed Duration: 400.000s",
                    "[Experiment 'ablation' Finished]",
                    "  Elapsed Duration: 280.000s",
                    "[Experiment 'stress' Finished]",
                    "  Elapsed Duration: 83.000s",
                    "  Pipeline Status: completed",
                ]
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(final_runtime_budget, "LOG_PATHS", (debug_log, small_log))
    monkeypatch.setattr(
        "sys.argv",
        [
            "final_runtime_budget.py",
            "--calibration-dir",
            str(tmp_path / "calibration"),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert final_runtime_budget.main() == 0
    assert (output_dir / "current_final_estimate.json").exists()
    assert (output_dir / "budget_candidates.json").exists()
    assert (output_dir / "runtime_summary.md").exists()
    payload = json.loads((output_dir / "current_final_estimate.json").read_text(encoding="utf-8"))
    assert "experiments" in payload
