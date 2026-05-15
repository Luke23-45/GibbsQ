#!/usr/bin/env python3
"""
Run verification support checks that are not publication-facing experiments.

This runner isolates operational and consistency checks from the core
paper-facing verification capsules so heavyweight checks such as engine
parity can be invoked explicitly without being mixed into the main
verification experiment rollup.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studies.runners.run_verification import (
    ExperimentResult,
    _run_experiment,
    _write_report,
    build_verification_check_experiments,
)

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "outputs/final"
DEFAULT_REPORT_DIR = "outputs/reports"
DEFAULT_CONFIG_NAME = "final_experiment"


def configure_logging() -> None:
    """Configure structured logging for the checks runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][verification-checks] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run verification support checks that are not publication-facing "
            "experiments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for check artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        default=DEFAULT_REPORT_DIR,
        help="Output directory for pipeline reports.",
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List checks without executing them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the verification checks runner."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    experiments = build_verification_check_experiments(args.output_dir, args.config_name)

    if args.dry_run:
        log.info("DRY RUN - %d support checks would be executed:", len(experiments))
        for name, hyp, _ in experiments:
            log.info("  - %s (%s)", name, hyp)
        return 0

    log.info("Starting verification support checks: %d items", len(experiments))
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
    log.info("  VERIFICATION SUPPORT CHECKS COMPLETE")
    log.info("  Pass: %d / %d  |  Fail: %d  |  Wall time: %.1fs",
             n_pass, len(results), n_fail, total_elapsed)
    log.info("=" * 70)

    report_path = _write_report(results, args.report_dir, total_elapsed)
    log.info("Report: %s", report_path)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
