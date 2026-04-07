from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.verification.runtime_budgeting import (
    ALL_GROUPED_EXPERIMENTS,
    latest_complete_sections_by_config,
)

LOG_PATHS = (
    Path("docs/logs/logs.md"),
    Path("docs/logs/logs1.md"),
    Path("docs/logs/logs2.md"),
    Path("docs/logs/logs3.md"),
    Path("docs/logs/logs4.md"),
)

STANDALONE_EXPERIMENTS = ("reinforce_train", "stress", "generalize", "critical", "ablation")


def build_runtime_plan(
    *,
    calibration_dir: Path,
    standalone_budget_minutes: float = 240.0,
) -> tuple[dict, dict[str, list[dict[str, float | str | bool]]], str]:
    sections = latest_complete_sections_by_config(LOG_PATHS)
    experiments: dict[str, dict[str, float | str | bool | None]] = {}
    for experiment in ALL_GROUPED_EXPERIMENTS:
        samples = []
        anchors = []
        for config_name, section in sections.items():
            if experiment in section.timings:
                samples.append(float(section.timings[experiment]))
                anchors.append(config_name)
        predicted_seconds = sum(samples) / len(samples) if samples else 0.0
        experiments[experiment] = {
            "predicted_seconds": predicted_seconds,
            "sample_count": len(samples),
            "anchor_configs": anchors,
            "standalone": experiment in STANDALONE_EXPERIMENTS,
        }

    current = {
        "calibration_dir": str(calibration_dir),
        "standalone_budget_minutes": float(standalone_budget_minutes),
        "experiments": experiments,
    }
    candidates = {
        experiment: [
            {
                "name": "current_estimate",
                "predicted_seconds": float(payload["predicted_seconds"]),
                "within_budget": float(payload["predicted_seconds"]) <= standalone_budget_minutes * 60.0,
            }
        ]
        for experiment, payload in experiments.items()
        if experiment in STANDALONE_EXPERIMENTS
    }
    summary = _render_summary(current, candidates)
    return current, candidates, summary


def _render_summary(current: dict, candidates: dict[str, list[dict[str, float | str | bool]]]) -> str:
    lines = [
        "# Runtime Summary",
        "",
        "## Current Final Estimate",
        "",
    ]
    for experiment, payload in current["experiments"].items():
        lines.append(
            f"- `{experiment}`: {float(payload['predicted_seconds']):.1f}s from {int(payload['sample_count'])} anchor(s)"
        )
    lines.extend(["", "## Budget Candidates", ""])
    for experiment, rows in candidates.items():
        for row in rows:
            lines.append(
                f"- `{experiment}` / {row['name']}: {float(row['predicted_seconds']):.1f}s "
                f"(within budget: {bool(row['within_budget'])})"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a runtime estimate for the final experiment campaign")
    parser.add_argument("--calibration-dir", type=Path, default=Path("outputs/runtime_calibration"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runtime_budget"))
    parser.add_argument("--standalone-budget-minutes", type=float, default=240.0)
    args = parser.parse_args()

    current, candidates, summary = build_runtime_plan(
        calibration_dir=args.calibration_dir,
        standalone_budget_minutes=args.standalone_budget_minutes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "current_final_estimate.json").write_text(
        json.dumps(current, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "budget_candidates.json").write_text(
        json.dumps(candidates, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "runtime_summary.md").write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
