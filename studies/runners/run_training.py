#!/usr/bin/env python3
"""
Run all z2 neural training experiments.

This runner executes the training phase (Phase 2) of the N-GibbsQ
neural learning pipeline. It runs Platinum BC Pretraining followed
by REINFORCE SSA Training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from studies.runners.common import PROJECT_ROOT, launch_module, resolve_runner_output_dir

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = "outputs/reports"
DEFAULT_CONFIG_NAME = "final_experiment"


@dataclass
class ExperimentResult:
    """Structured result for a single training execution."""

    name: str
    hypothesis: str
    module: str
    status: str = "NOT_RUN"
    elapsed_seconds: float = 0.0
    output_files: list[str] = field(default_factory=list)
    error_message: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "module": self.module,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "output_files": self.output_files,
            "error_message": self.error_message,
        }


def _experiment_type_for_module(module: str) -> str:
    mapping = {
        "gibbsq.experiments.training.pretrain_bc": "bc_train",
        "gibbsq.experiments.training.train_reinforce": "reinforce_train",
    }
    return mapping[module]


def _run_module(
    name: str,
    hypothesis: str,
    module: str,
    config_name: str,
    output_dir: str | None,
) -> ExperimentResult:
    """Execute a single training module."""
    result = ExperimentResult(name=name, hypothesis=hypothesis, module=module)
    log.info("=" * 70)
    log.info("  EXPERIMENT: %s  (supports %s)", name, hypothesis)
    log.info("=" * 70)

    t0 = time.perf_counter()
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    try:
        output_files = launch_module(
            module=module,
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=True,
            experiment_type=_experiment_type_for_module(module),
        )
        result.elapsed_seconds = time.perf_counter() - t0
        result.status = "PASS"
        result.output_files = [str(path) for path in output_files]
        log.info("  OK %s completed in %.2fs", name, result.elapsed_seconds)
    except Exception as exc:
        result.elapsed_seconds = time.perf_counter() - t0
        result.status = "FAIL"
        result.error_message = str(exc)
        result.traceback = traceback.format_exc()
        log.error("  FAIL %s after %.2fs: %s", name, result.elapsed_seconds, exc)
    return result


def build_training_experiments() -> list[tuple[str, str, str]]:
    """Build the ordered list of training experiments."""
    return [
        (
            "Platinum BC Pretraining",
            "H7",
            "gibbsq.experiments.training.pretrain_bc",
        ),
        (
            "REINFORCE SSA Training",
            "H7",
            "gibbsq.experiments.training.train_reinforce",
        ),
    ]


def _write_report(
    results: list[ExperimentResult],
    report_dir: str | Path,
    total_elapsed: float,
) -> Path:
    """Write structured training pipeline report."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    json_report = {
        "pipeline": "training",
        "timestamp": timestamp,
        "overall_status": overall,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "summary": {
            "total": len(results),
            "pass": n_pass,
            "fail": n_fail,
        },
        "experiments": [r.to_dict() for r in results],
    }
    json_path = report_dir / f"training_report_{timestamp}.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    lines = [
        f"# Training Pipeline Report - {timestamp}",
        "",
        f"**Overall**: {overall}  |  "
        f"**Pass**: {n_pass}  |  **Fail**: {n_fail}  |  "
        f"**Wall time**: {total_elapsed:.1f}s",
        "",
        "| # | Experiment | Hypothesis | Module | Status | Time (s) | Output Files |",
        "|---|-----------|-----------|--------|--------|----------|-------------|",
    ]
    for idx, r in enumerate(results, start=1):
        files_str = ", ".join(Path(f).name for f in r.output_files) or "-"
        lines.append(
            f"| {idx} | {r.name} | {r.hypothesis} | "
            f"`{r.module}` | {r.status} | {r.elapsed_seconds:.1f} | {files_str} |"
        )
    if n_fail > 0:
        lines.extend(["", "## Failures", ""])
        for r in results:
            if r.status == "FAIL":
                lines.append(f"### {r.name}")
                lines.append(f"```\n{r.error_message}\n```")
    lines.append("")

    md_path = report_dir / f"training_report_{timestamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][training] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run z2 neural training experiments (Phase 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root for experiment capsules. Defaults to the selected config's output_dir.",
    )
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    experiments = build_training_experiments()

    if args.dry_run:
        log.info("DRY RUN - %d experiments would be executed:", len(experiments))
        for name, hyp, module in experiments:
            log.info("  - %s (%s) -> %s", name, hyp, module)
        return 0

    log.info("Starting training pipeline: %d experiments", len(experiments))
    t_pipeline = time.perf_counter()

    results: list[ExperimentResult] = []
    for name, hypothesis, module in experiments:
        result = _run_module(name, hypothesis, module, args.config_name, args.output_dir)
        results.append(result)
        if result.status == "FAIL":
            log.error("Pipeline halted due to failure in %s", name)
            break

    total_elapsed = time.perf_counter() - t_pipeline

    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")

    log.info("")
    log.info("=" * 70)
    log.info("  TRAINING PIPELINE COMPLETE")
    log.info("  Pass: %d / %d  |  Fail: %d  |  Wall time: %.1fs",
             n_pass, len(experiments), n_fail, total_elapsed)
    log.info("=" * 70)

    report_path = _write_report(results, args.report_dir, total_elapsed)
    log.info("Report: %s", report_path)

    return 0 if n_fail == 0 and len(results) == len(experiments) else 1


if __name__ == "__main__":
    raise SystemExit(main())
