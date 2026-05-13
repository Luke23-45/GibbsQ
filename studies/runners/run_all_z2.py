#!/usr/bin/env python3
"""
Master runner: execute the complete z2 experiment pipeline.

This script runs verification experiments followed by benchmark
experiments, producing a unified pipeline report at the end.

Usage:
    python -m studies.runners.run_all_z2
    python -m studies.runners.run_all_z2 --skip-benchmarks
    python -m studies.runners.run_all_z2 --dry-run

The verification stage covers H1–H4 and H3 (deterministic theory,
drift audit, boundary mismatch).  The benchmark stage covers H5
(independent-seed policy rerun).

Design:
    - Verification runs first because it is fast (< 2 min).
    - Benchmarks run second because they are slow (~30 min).
    - --skip-benchmarks allows a quick verification-only run.
    - The exit code is 0 only if ALL stages pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studies.runners.run_verification import (
    ExperimentResult,
    _run_experiment,
    build_verification_experiments,
)
from studies.runners.run_benchmarks import (
    build_benchmark_experiments,
)

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "outputs/data"
DEFAULT_REPORT_DIR = "outputs/reports"


def _write_master_report(
    verification_results: list[ExperimentResult],
    benchmark_results: list[ExperimentResult],
    report_dir: str | Path,
    total_elapsed: float,
    *,
    benchmarks_skipped: bool = False,
) -> Path:
    """Write the unified master pipeline report."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_results = verification_results + benchmark_results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_pass = sum(1 for r in all_results if r.status == "PASS")
    n_fail = sum(1 for r in all_results if r.status == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    # ── JSON ──
    json_report = {
        "pipeline": "master_z2",
        "timestamp": timestamp,
        "overall_status": overall,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "benchmarks_skipped": benchmarks_skipped,
        "summary": {
            "total": len(all_results),
            "pass": n_pass,
            "fail": n_fail,
        },
        "verification": [r.to_dict() for r in verification_results],
        "benchmark": [r.to_dict() for r in benchmark_results],
    }
    json_path = report_dir / f"z2_pipeline_report_{timestamp}.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    # ── Markdown ──
    lines = [
        f"# z2 Master Pipeline Report — {timestamp}",
        "",
        f"**Overall**: {overall}  |  "
        f"**Pass**: {n_pass}/{len(all_results)}  |  "
        f"**Fail**: {n_fail}  |  "
        f"**Wall time**: {total_elapsed:.1f}s",
        "",
        "## Verification Stage (H1–H4, H3)",
        "",
        "| # | Experiment | Hypothesis | Status | Time (s) |",
        "|---|-----------|-----------|--------|----------|",
    ]
    for idx, r in enumerate(verification_results, start=1):
        lines.append(
            f"| {idx} | {r.name} | {r.hypothesis} | "
            f"{r.status} | {r.elapsed_seconds:.1f} |"
        )

    if benchmarks_skipped:
        lines.extend(["", "## Benchmark Stage (skipped)", ""])
    else:
        lines.extend([
            "",
            "## Benchmark Stage (H5)",
            "",
            "| # | Experiment | Hypothesis | Status | Time (s) |",
            "|---|-----------|-----------|--------|----------|",
        ])
        for idx, r in enumerate(benchmark_results, start=1):
            lines.append(
                f"| {idx} | {r.name} | {r.hypothesis} | "
                f"{r.status} | {r.elapsed_seconds:.1f} |"
            )

    if n_fail > 0:
        lines.extend(["", "## Failures", ""])
        for r in all_results:
            if r.status == "FAIL":
                lines.append(f"### {r.name}")
                lines.append(f"```\n{r.error_message}\n```")
                lines.append("")

    lines.append("")
    md_path = report_dir / f"z2_pipeline_report_{timestamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][z2] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Master runner: execute the complete z2 experiment pipeline. "
            "Runs verification first (fast), then benchmarks (slow)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Run only verification experiments (skip slow benchmark simulations).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    verification_exps = build_verification_experiments(args.output_dir)
    benchmark_exps = build_benchmark_experiments(args.output_dir)

    if args.dry_run:
        log.info("DRY RUN — Pipeline would execute:")
        log.info("  Verification (%d experiments):", len(verification_exps))
        for name, hyp, _ in verification_exps:
            log.info("    • %s (%s)", name, hyp)
        if args.skip_benchmarks:
            log.info("  Benchmarks: SKIPPED")
        else:
            log.info("  Benchmarks (%d experiments):", len(benchmark_exps))
            for name, hyp, _ in benchmark_exps:
                log.info("    • %s (%s)", name, hyp)
        return 0

    log.info("=" * 70)
    log.info("  z2 MASTER PIPELINE")
    log.info("  Verification: %d experiments", len(verification_exps))
    log.info("  Benchmark:    %d experiments%s",
             len(benchmark_exps), " (SKIPPED)" if args.skip_benchmarks else "")
    log.info("=" * 70)

    t_pipeline = time.perf_counter()

    # ── Stage 1: Verification ──
    log.info("")
    log.info("━" * 70)
    log.info("  STAGE 1: VERIFICATION")
    log.info("━" * 70)

    verification_results: list[ExperimentResult] = []
    for name, hypothesis, func in verification_exps:
        result = _run_experiment(name, hypothesis, func)
        verification_results.append(result)

    # ── Stage 2: Benchmarks ──
    benchmark_results: list[ExperimentResult] = []
    if not args.skip_benchmarks:
        log.info("")
        log.info("━" * 70)
        log.info("  STAGE 2: BENCHMARKS")
        log.info("━" * 70)

        for name, hypothesis, func in benchmark_exps:
            result = _run_experiment(name, hypothesis, func)
            benchmark_results.append(result)

    total_elapsed = time.perf_counter() - t_pipeline

    # ── Summary ──
    all_results = verification_results + benchmark_results
    n_pass = sum(1 for r in all_results if r.status == "PASS")
    n_fail = sum(1 for r in all_results if r.status == "FAIL")

    log.info("")
    log.info("=" * 70)
    log.info("  z2 PIPELINE COMPLETE")
    log.info("  Pass: %d / %d  |  Fail: %d  |  Wall time: %.1fs",
             n_pass, len(all_results), n_fail, total_elapsed)
    log.info("=" * 70)

    report_path = _write_master_report(
        verification_results, benchmark_results,
        args.report_dir, total_elapsed,
        benchmarks_skipped=args.skip_benchmarks,
    )
    log.info("Report: %s", report_path)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
