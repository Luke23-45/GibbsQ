#!/usr/bin/env python3
"""
Run all z2 benchmark experiments.

This runner executes the benchmark-level experiments that support
Hypothesis H5 (empirical benchmark performance).

Usage:
    python -m studies.runners.run_benchmarks
    python -m studies.runners.run_benchmarks --output-dir outputs/data
    python -m studies.runners.run_benchmarks --dry-run
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
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "outputs/data"
DEFAULT_REPORT_DIR = "outputs/reports"


@dataclass
class ExperimentResult:
    """Structured result for a single experiment execution."""

    name: str
    hypothesis: str
    status: str = "NOT_RUN"
    elapsed_seconds: float = 0.0
    output_files: list[str] = field(default_factory=list)
    error_message: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "output_files": self.output_files,
            "error_message": self.error_message,
        }


def _run_experiment(
    name: str,
    hypothesis: str,
    func: Callable[[], list[Path | str]],
) -> ExperimentResult:
    """Execute a single experiment with full error isolation."""
    result = ExperimentResult(name=name, hypothesis=hypothesis)
    log.info("=" * 70)
    log.info("  EXPERIMENT: %s  (supports %s)", name, hypothesis)
    log.info("=" * 70)

    t0 = time.perf_counter()
    try:
        output_files = func()
        result.elapsed_seconds = time.perf_counter() - t0
        result.output_files = [str(f) for f in output_files]
        result.status = "PASS"
        log.info(
            "  ✓ %s completed in %.2fs — %d output files",
            name, result.elapsed_seconds, len(result.output_files),
        )
    except Exception as exc:
        result.elapsed_seconds = time.perf_counter() - t0
        result.status = "FAIL"
        result.error_message = str(exc)
        result.traceback = traceback.format_exc()
        log.error(
            "  ✗ %s FAILED after %.2fs: %s",
            name, result.elapsed_seconds, exc,
        )
        log.error("  Traceback:\n%s", result.traceback)

    return result


# ──────────────────────────────────────────────────────────────────────
# Experiment definitions
# ──────────────────────────────────────────────────────────────────────

def _make_independent_seed_rerun(output_dir: str) -> Callable[[], list[Path]]:
    """Create a callable for the independent-seed benchmark rerun."""
    def run() -> list[Path]:
        from gibbsq.experiments.benchmark.independent_seed_rerun import (
            run_benchmark_rerun,
        )
        policy_path, comp_path = run_benchmark_rerun(output_dir)
        return [policy_path, comp_path]
    return run


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def _write_report(
    results: list[ExperimentResult],
    report_dir: str | Path,
    total_elapsed: float,
) -> Path:
    """Write structured pipeline report."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    json_report = {
        "pipeline": "benchmark",
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
    json_path = report_dir / f"benchmark_report_{timestamp}.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    lines = [
        f"# Benchmark Pipeline Report — {timestamp}",
        "",
        f"**Overall**: {overall}  |  "
        f"**Pass**: {n_pass}  |  **Fail**: {n_fail}  |  "
        f"**Wall time**: {total_elapsed:.1f}s",
        "",
        "| # | Experiment | Hypothesis | Status | Time (s) |",
        "|---|-----------|-----------|--------|----------|",
    ]
    for idx, r in enumerate(results, start=1):
        lines.append(
            f"| {idx} | {r.name} | {r.hypothesis} | "
            f"{r.status} | {r.elapsed_seconds:.1f} |"
        )
    if n_fail > 0:
        lines.extend(["", "## Failures", ""])
        for r in results:
            if r.status == "FAIL":
                lines.append(f"### {r.name}")
                lines.append(f"```\n{r.error_message}\n```")
    lines.append("")

    md_path = report_dir / f"benchmark_report_{timestamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def build_benchmark_experiments(output_dir: str) -> list[tuple[str, str, Callable]]:
    """Build the ordered list of benchmark experiments."""
    return [
        (
            "Independent-Seed Benchmark Rerun",
            "H5",
            _make_independent_seed_rerun(output_dir),
        ),
    ]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][benchmark] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run z2 benchmark experiments (H5). "
            "NOTE: The independent-seed rerun is simulation-heavy "
            "and may take ~30 minutes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    experiments = build_benchmark_experiments(args.output_dir)

    if args.dry_run:
        log.info("DRY RUN — %d experiments would be executed:", len(experiments))
        for name, hyp, _ in experiments:
            log.info("  • %s (%s)", name, hyp)
        return 0

    log.info("Starting benchmark pipeline: %d experiments", len(experiments))
    t_pipeline = time.perf_counter()

    results: list[ExperimentResult] = []
    for name, hypothesis, func in experiments:
        result = _run_experiment(name, hypothesis, func)
        results.append(result)

    total_elapsed = time.perf_counter() - t_pipeline

    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")

    log.info("")
    log.info("=" * 70)
    log.info("  BENCHMARK PIPELINE COMPLETE")
    log.info("  Pass: %d / %d  |  Fail: %d  |  Wall time: %.1fs",
             n_pass, len(results), n_fail, total_elapsed)
    log.info("=" * 70)

    report_path = _write_report(results, args.report_dir, total_elapsed)
    log.info("Report: %s", report_path)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
