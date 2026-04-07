from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

GROUP_A_EXPERIMENTS = ["check_configs", "reinforce_check", "drift", "sweep"]
GROUP_B_EXPERIMENTS = ["stress", "bc_train", "reinforce_train", "policy"]
GROUP_C_EXPERIMENTS = ["stats", "generalize", "critical", "ablation"]
ALL_GROUPED_EXPERIMENTS = GROUP_A_EXPERIMENTS + GROUP_B_EXPERIMENTS + GROUP_C_EXPERIMENTS

_START_RE = re.compile(
    r"scripts/execution/(?:reproduction_pipeline|experiment_runner)\.py(?:\s+--config-name\s+(?P<cfg>\S+))?"
)
_FINISHED_RE = re.compile(r"\[Experiment '([^']+)' Finished\]")
_ELAPSED_RE = re.compile(r"Elapsed Duration:\s*([0-9]+(?:\.[0-9]+)?)s")
_PIPELINE_STATUS_RE = re.compile(r"Pipeline Status:\s*(.+)")


@dataclass(frozen=True)
class CompleteSection:
    config_name: str
    timings: dict[str, float]
    completed: bool
    source_path: Path


def default_calibration_matrix(config_name: str, experiment_name: str) -> list[dict[str, object]]:
    if experiment_name == "reinforce_train":
        return [
            {"probe_case": "baseline"},
            {"probe_case": "batch", "batch_size": 8 if config_name == "small" else 16},
            {"probe_case": "eval", "eval_batches": 1, "eval_trajs_per_batch": 1},
            {"probe_case": "sim_time", "sim_time": 250.0 if config_name == "small" else 500.0},
        ]
    return [{"probe_case": "baseline"}]


def default_probe_overrides(config_name: str, experiment_name: str) -> list[str]:
    overrides = ["train_epochs=1"]
    if experiment_name == "reinforce_train":
        overrides.extend(
            [
                "neural_training.eval_batches=1",
                "neural_training.eval_trajs_per_batch=1",
            ]
        )
    return overrides


def latest_complete_sections_by_config(log_paths: list[Path] | tuple[Path, ...]) -> dict[str, CompleteSection]:
    latest: dict[str, CompleteSection] = {}
    for log_path in log_paths:
        if not Path(log_path).exists():
            continue
        sections = _parse_complete_sections(Path(log_path))
        for section in sections:
            latest[section.config_name] = section
    return latest


def _parse_complete_sections(log_path: Path) -> list[CompleteSection]:
    sections: list[CompleteSection] = []
    current_config: str | None = None
    current_timings: dict[str, float] = {}
    current_experiment: str | None = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        start_match = _START_RE.search(line)
        if start_match:
            current_config = start_match.group("cfg") or "default"
            current_timings = {}
            current_experiment = None
            continue

        finished_match = _FINISHED_RE.search(line)
        if finished_match:
            current_experiment = finished_match.group(1)
            continue

        elapsed_match = _ELAPSED_RE.search(line)
        if elapsed_match and current_config and current_experiment:
            current_timings[current_experiment] = float(elapsed_match.group(1))
            current_experiment = None
            continue

        status_match = _PIPELINE_STATUS_RE.search(line)
        if status_match and current_config:
            status_text = status_match.group(1).strip().lower()
            completed = "completed" in status_text and "fail" not in status_text and "interrupt" not in status_text
            if completed:
                sections.append(
                    CompleteSection(
                        config_name=current_config,
                        timings=dict(current_timings),
                        completed=True,
                        source_path=log_path,
                    )
                )
            current_config = None
            current_timings = {}
            current_experiment = None

    return sections

