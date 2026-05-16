#!/usr/bin/env python3
"""
Master runner: execute the complete z2 experiment pipeline.

This runner supports three isolated phases plus a full end-to-end mode.
Each phase owns a disjoint subset of experiments so users can stop after
the selected slice without accidentally running the rest of the pipeline.

Usage:
    python -m studies.runners.run_all_z2
    python -m studies.runners.run_all_z2 --phase phase1
    python -m studies.runners.run_all_z2 --phase phase2
    python -m studies.runners.run_all_z2 --phase phase3
    python -m studies.runners.run_all_z2 --phase full --skip-benchmarks
    python -m studies.runners.run_all_z2 --dry-run

Phase layout:
    - phase1: fastest deterministic / light verification experiments
    - phase2: heavier stochastic verification experiments
    - phase3: slow benchmark rerun
    - full: phase1 -> phase2 -> phase3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studies.runners.run_benchmarks import build_benchmark_experiments
from studies.runners.run_verification import (
    ExperimentResult,
    _run_experiment,
    build_verification_experiments,
)

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = None
DEFAULT_REPORT_DIR = "outputs/reports"

ExperimentSpec = tuple[str, str, Callable[[], list[Path | str]]]

PHASE_ORDER = ("phase1", "phase2", "phase3")
PHASE_TITLES = {
    "phase1": "PHASE 1: FAST VERIFICATION",
    "phase2": "PHASE 2: HEAVIER VERIFICATION",
    "phase3": "PHASE 3: BENCHMARKS",
}
EXPERIMENT_PHASES = {
    "Boundary Equilibrium Verification": "phase1",
    "Reflected-ODE Multi-Start Convergence": "phase1",
    "Theorem-Constant Parameter Sweep": "phase1",
    "CTMC Generator Boundary-Mismatch Demo": "phase2",
    "Direct CTMC Validation Capsule": "phase2",
    "Exhaustive Small-Grid Drift Audit": "phase2",
    "Consolidated CTMC Support Summary": "phase2",
    "Independent-Seed Benchmark Rerun": "phase3",
}


def build_phase_experiments(
    output_dir: str | None,
    config_name: str,
) -> dict[str, list[ExperimentSpec]]:
    """Partition the full z2 experiment catalog into execution phases."""
    verification_experiments = build_verification_experiments(output_dir, config_name)
    benchmark_experiments = build_benchmark_experiments(output_dir, config_name)
    all_experiments = [*verification_experiments, *benchmark_experiments]

    phase_map = {phase: [] for phase in PHASE_ORDER}
    missing_phase_assignments: list[str] = []

    for experiment in all_experiments:
        name = experiment[0]
        phase = EXPERIMENT_PHASES.get(name)
        if phase is None:
            missing_phase_assignments.append(name)
            continue
        phase_map[phase].append(experiment)

    if missing_phase_assignments:
        raise ValueError(
            "Missing phase assignments for experiments: "
            + ", ".join(missing_phase_assignments)
        )

    assigned_names = {name for experiments in phase_map.values() for name, _, _ in experiments}
    extra_assignments = sorted(set(EXPERIMENT_PHASES) - assigned_names)
    if extra_assignments:
        raise ValueError(
            "Phase catalog includes experiments not present in run_all_z2: "
            + ", ".join(extra_assignments)
        )

    return phase_map


def _write_master_report(
    phase_results: dict[str, list[ExperimentResult]],
    report_dir: str | Path,
    total_elapsed: float,
    *,
    phase_selection: str,
    benchmarks_skipped: bool = False,
) -> Path:
    """Write the unified master pipeline report."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    executed_phases = [phase for phase in PHASE_ORDER if phase in phase_results]
    all_results = [
        result
        for phase in executed_phases
        for result in phase_results[phase]
    ]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_pass = sum(1 for result in all_results if result.status == "PASS")
    n_fail = sum(1 for result in all_results if result.status == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    json_report = {
        "pipeline": "master_z2",
        "timestamp": timestamp,
        "phase_selection": phase_selection,
        "executed_phases": executed_phases,
        "overall_status": overall,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "benchmarks_skipped": benchmarks_skipped,
        "summary": {
            "total": len(all_results),
            "pass": n_pass,
            "fail": n_fail,
        },
        "phases": {
            phase: [result.to_dict() for result in phase_results[phase]]
            for phase in executed_phases
        },
    }
    json_path = report_dir / f"z2_pipeline_report_{timestamp}.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    lines = [
        f"# z2 Master Pipeline Report - {timestamp}",
        "",
        f"**Overall**: {overall}  |  "
        f"**Pass**: {n_pass}/{len(all_results)}  |  "
        f"**Fail**: {n_fail}  |  "
        f"**Wall time**: {total_elapsed:.1f}s  |  "
        f"**Selection**: {phase_selection}",
        "",
    ]

    for phase in executed_phases:
        lines.extend([
            f"## {PHASE_TITLES[phase]}",
            "",
            "| # | Experiment | Hypothesis | Status | Time (s) |",
            "|---|-----------|-----------|--------|----------|",
        ])
        for idx, result in enumerate(phase_results[phase], start=1):
            lines.append(
                f"| {idx} | {result.name} | {result.hypothesis} | "
                f"{result.status} | {result.elapsed_seconds:.1f} |"
            )
        lines.append("")

    if benchmarks_skipped:
        lines.extend(["## Phase 3: Benchmarks (skipped)", ""])

    if n_fail > 0:
        lines.extend(["## Failures", ""])
        for result in all_results:
            if result.status == "FAIL":
                lines.append(f"### {result.name}")
                lines.append(f"```\n{result.error_message}\n```")
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
            "Master runner: execute the z2 experiment pipeline. "
            "Use --phase phase1/phase2/phase3 for isolated slices or "
            "--phase full for the whole sequence."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output root for experiment capsules. Defaults to the selected config's output_dir.",
    )
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--config-name", default="final_experiment")
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "phase3", "full"],
        default="full",
        help=(
            "Execution slice to run. phase1/phase2/phase3 are isolated; "
            "full runs all three phases in order."
        ),
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="With --phase full, stop after phase2 and skip phase3 benchmarks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    phase_experiments = build_phase_experiments(args.output_dir, args.config_name)

    if args.phase == "phase3" and args.skip_benchmarks:
        parser.error("--skip-benchmarks cannot be combined with --phase phase3.")

    if args.phase == "full":
        selected_phases = list(PHASE_ORDER[:-1] if args.skip_benchmarks else PHASE_ORDER)
    else:
        selected_phases = [args.phase]

    if args.dry_run:
        log.info("DRY RUN - Pipeline would execute:")
        log.info("  Phase selection: %s", args.phase)
        for phase in selected_phases:
            experiments = phase_experiments[phase]
            log.info("  %s (%d experiments):", PHASE_TITLES[phase], len(experiments))
            for name, hypothesis, _ in experiments:
                log.info("    - %s (%s)", name, hypothesis)
        if args.skip_benchmarks and args.phase == "full":
            log.info("  Phase 3 benchmarks: SKIPPED")
        return 0

    log.info("=" * 70)
    log.info("  z2 MASTER PIPELINE")
    log.info("  Phase selection: %s", args.phase)
    for phase in selected_phases:
        log.info("  %s -> %d experiments", phase, len(phase_experiments[phase]))
    if args.skip_benchmarks and args.phase == "full":
        log.info("  phase3 -> SKIPPED")
    log.info("=" * 70)

    t_pipeline = time.perf_counter()
    phase_results: dict[str, list[ExperimentResult]] = {}

    for phase in selected_phases:
        log.info("")
        log.info("-" * 70)
        log.info("  %s", PHASE_TITLES[phase])
        log.info("-" * 70)

        results: list[ExperimentResult] = []
        for name, hypothesis, func in phase_experiments[phase]:
            result = _run_experiment(name, hypothesis, func)
            results.append(result)
        phase_results[phase] = results

    total_elapsed = time.perf_counter() - t_pipeline
    all_results = [
        result
        for phase in selected_phases
        for result in phase_results[phase]
    ]
    n_pass = sum(1 for result in all_results if result.status == "PASS")
    n_fail = sum(1 for result in all_results if result.status == "FAIL")

    log.info("")
    log.info("=" * 70)
    log.info("  z2 PIPELINE COMPLETE")
    log.info(
        "  Pass: %d / %d  |  Fail: %d  |  Wall time: %.1fs",
        n_pass,
        len(all_results),
        n_fail,
        total_elapsed,
    )
    log.info("=" * 70)

    report_path = _write_master_report(
        phase_results,
        args.report_dir,
        total_elapsed,
        phase_selection=args.phase,
        benchmarks_skipped=args.skip_benchmarks and args.phase == "full",
    )
    log.info("Report: %s", report_path)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
