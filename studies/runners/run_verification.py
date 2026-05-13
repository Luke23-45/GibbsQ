#!/usr/bin/env python3
"""
Run all z2 verification experiments.

This runner executes every verification experiment that supports the
z2 thesis hypothesis ladder (H1–H4, H3).  Each experiment runs in its
own isolated error-handling block so that a single failure does not
abort the pipeline.

Usage:
    python -m studies.runners.run_verification
    python -m studies.runners.run_verification --output-dir outputs/data
    python -m studies.runners.run_verification --dry-run

Design:
    - Every experiment call is wrapped in _run_experiment() which
      captures timing, stdout/stderr, and exception information.
    - Results are collected into a structured report written at the end.
    - Exit code is 0 only if ALL experiments pass.
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


# ──────────────────────────────────────────────────────────────────────
# Result tracking
# ──────────────────────────────────────────────────────────────────────

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
        """Convert to JSON-serializable dictionary."""
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
    """Execute a single experiment with full error isolation.

    Parameters
    ----------
    name : str
        Human-readable experiment name.
    hypothesis : str
        Thesis hypothesis this experiment supports.
    func : callable
        Zero-argument callable that runs the experiment and returns
        a list of output file paths.

    Returns
    -------
    ExperimentResult
        Structured result with timing and error information.
    """
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

def _make_boundary_equilibrium(output_dir: str) -> Callable[[], list[Path]]:
    """Create a callable for the boundary-equilibrium experiment."""
    def run() -> list[Path]:
        from gibbsq.experiments.verification.boundary_equilibrium_verification import (
            benchmark_systems,
            run_verification,
        )
        csv_path = run_verification(benchmark_systems(), output_dir)
        return [csv_path]
    return run


def _make_reflected_ode_convergence(output_dir: str) -> Callable[[], list[Path]]:
    """Create a callable for the reflected-ODE convergence experiment."""
    def run() -> list[Path]:
        from gibbsq.experiments.verification.reflected_ode_convergence import (
            benchmark_systems,
            run_convergence_verification,
        )
        traj_path, summary_path = run_convergence_verification(
            benchmark_systems(), output_dir,
        )
        return [traj_path, summary_path]
    return run


def _make_exhaustive_drift_audit(output_dir: str) -> Callable[[], list[Path]]:
    """Create a callable for the exhaustive drift audit experiment."""
    def run() -> list[Path]:
        from gibbsq.experiments.verification.exhaustive_drift_audit import (
            run_exhaustive_audit,
            toy_systems,
        )
        grid_path, summary_path = run_exhaustive_audit(
            toy_systems(), output_dir, max_norm=30,
        )
        return [grid_path, summary_path]
    return run


def _make_theorem_constant_sweep(output_dir: str) -> Callable[[], list[Path]]:
    """Create a callable for the theorem-constant sweep experiment."""
    def run() -> list[Path]:
        from gibbsq.experiments.verification.theorem_constant_sweep import (
            run_theorem_constant_sweep,
        )
        csv_path = run_theorem_constant_sweep(output_dir)
        return [csv_path]
    return run


def _make_boundary_mismatch_demo(output_dir: str) -> Callable[[], list[Path]]:
    """Create a callable for the CTMC boundary-mismatch experiment."""
    def run() -> list[Path]:
        from gibbsq.experiments.verification.ctmc_boundary_mismatch_demo import (
            demo_systems,
            run_boundary_mismatch_demo,
        )
        state_path, summary_path = run_boundary_mismatch_demo(
            demo_systems(), output_dir, max_norm=15,
        )
        return [state_path, summary_path]
    return run


# ──────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────

def _write_report(
    results: list[ExperimentResult],
    report_dir: str | Path,
    total_elapsed: float,
) -> Path:
    """Write a structured pipeline report.

    Writes both a JSON report (machine-readable) and a Markdown report
    (human-readable).

    Parameters
    ----------
    results : list of ExperimentResult
        Results from each experiment.
    report_dir : str or Path
        Output directory for reports.
    total_elapsed : float
        Total pipeline wall time in seconds.

    Returns
    -------
    Path
        Path to the Markdown report.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "NOT_RUN")
    overall = "PASS" if n_fail == 0 and n_skip == 0 else "FAIL"

    # ── JSON report ──
    json_report = {
        "pipeline": "verification",
        "timestamp": timestamp,
        "overall_status": overall,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "summary": {
            "total": len(results),
            "pass": n_pass,
            "fail": n_fail,
            "not_run": n_skip,
        },
        "experiments": [r.to_dict() for r in results],
    }
    json_path = report_dir / f"verification_report_{timestamp}.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    # ── Markdown report ──
    lines = [
        f"# Verification Pipeline Report — {timestamp}",
        "",
        f"**Overall**: {overall}  |  "
        f"**Pass**: {n_pass}  |  **Fail**: {n_fail}  |  "
        f"**Skipped**: {n_skip}  |  "
        f"**Wall time**: {total_elapsed:.1f}s",
        "",
        "| # | Experiment | Hypothesis | Status | Time (s) | Output Files |",
        "|---|-----------|-----------|--------|----------|-------------|",
    ]
    for idx, r in enumerate(results, start=1):
        files_str = ", ".join(Path(f).name for f in r.output_files) or "—"
        lines.append(
            f"| {idx} | {r.name} | {r.hypothesis} | "
            f"{r.status} | {r.elapsed_seconds:.1f} | {files_str} |"
        )

    if n_fail > 0:
        lines.extend(["", "## Failures", ""])
        for r in results:
            if r.status == "FAIL":
                lines.append(f"### {r.name}")
                lines.append(f"```\n{r.error_message}\n```")
                lines.append("")

    lines.append("")
    md_path = report_dir / f"verification_report_{timestamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    log.info("Reports written to: %s", report_dir)
    return md_path


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def build_verification_experiments(output_dir: str) -> list[tuple[str, str, Callable]]:
    """Build the ordered list of verification experiments.

    Returns
    -------
    list of (name, hypothesis, callable)
    """
    return [
        (
            "Boundary Equilibrium Verification",
            "H2",
            _make_boundary_equilibrium(output_dir),
        ),
        (
            "Reflected-ODE Multi-Start Convergence",
            "H1, H2",
            _make_reflected_ode_convergence(output_dir),
        ),
        (
            "Exhaustive Small-Grid Drift Audit",
            "H4",
            _make_exhaustive_drift_audit(output_dir),
        ),
        (
            "Theorem-Constant Parameter Sweep",
            "H4",
            _make_theorem_constant_sweep(output_dir),
        ),
        (
            "CTMC Generator Boundary-Mismatch Demo",
            "H3",
            _make_boundary_mismatch_demo(output_dir),
        ),
    ]


def configure_logging() -> None:
    """Configure structured logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][verification] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all z2 verification experiments (H1–H4, H3). "
            "Each experiment runs in isolation; failures are reported "
            "but do not abort the pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for experiment CSV files.",
    )
    parser.add_argument(
        "--report-dir",
        default=DEFAULT_REPORT_DIR,
        help="Output directory for pipeline reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List experiments without executing them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the verification runner."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    experiments = build_verification_experiments(args.output_dir)

    if args.dry_run:
        log.info("DRY RUN — %d experiments would be executed:", len(experiments))
        for name, hyp, _ in experiments:
            log.info("  • %s (%s)", name, hyp)
        return 0

    log.info("Starting verification pipeline: %d experiments", len(experiments))
    t_pipeline = time.perf_counter()

    results: list[ExperimentResult] = []
    for name, hypothesis, func in experiments:
        result = _run_experiment(name, hypothesis, func)
        results.append(result)

    total_elapsed = time.perf_counter() - t_pipeline

    # ── Summary ──
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")

    log.info("")
    log.info("=" * 70)
    log.info("  VERIFICATION PIPELINE COMPLETE")
    log.info("  Pass: %d / %d  |  Fail: %d  |  Wall time: %.1fs",
             n_pass, len(results), n_fail, total_elapsed)
    log.info("=" * 70)

    report_path = _write_report(results, args.report_dir, total_elapsed)
    log.info("Report: %s", report_path)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
