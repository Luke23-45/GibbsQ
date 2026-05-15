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

from studies.runners.common import (
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    launch_module,
    resolve_runner_output_dir,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = "outputs/reports"
DEFAULT_CONFIG_NAME = "final_experiment"


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

def _make_boundary_equilibrium(output_dir: str, config_name: str) -> Callable[[], list[Path]]:
    """Create a launcher for the boundary-equilibrium experiment."""
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.verification.boundary_equilibrium_verification",
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=False,
            experiment_type="boundary_equilibrium_verification",
        )
    return run


def _make_reflected_ode_convergence(output_dir: str, config_name: str) -> Callable[[], list[Path]]:
    """Create a launcher for the reflected-ODE convergence experiment."""
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.verification.reflected_ode_convergence",
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=False,
            experiment_type="reflected_ode_convergence",
        )
    return run


def _make_exhaustive_drift_audit(output_dir: str, config_name: str) -> Callable[[], list[Path]]:
    """Create a launcher for the exhaustive drift audit experiment."""
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.verification.exhaustive_drift_audit",
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=False,
            include_config_name=False,
            extra_args=["--max-norm", "30"],
            experiment_type="exhaustive_drift_audit",
        )
    return run


def _make_theorem_constant_sweep(output_dir: str, config_name: str) -> Callable[[], list[Path]]:
    """Create a launcher for the theorem-constant sweep experiment."""
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.verification.theorem_constant_sweep",
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=False,
            experiment_type="theorem_constant_sweep",
        )
    return run


def _make_boundary_mismatch_demo(output_dir: str, config_name: str) -> Callable[[], list[Path]]:
    """Create a launcher for the CTMC boundary-mismatch experiment."""
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.verification.ctmc_boundary_mismatch_demo",
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=False,
            include_config_name=False,
            extra_args=["--max-norm", "15"],
            experiment_type="ctmc_boundary_mismatch_demo",
        )
    return run


def _make_direct_ctmc_validation(output_dir: str, config_name: str) -> Callable[[], list[Path]]:
    """Create a launcher for the direct CTMC validation capsule."""
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.verification.direct_ctmc_validation",
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=False,
            extra_args=["--mode", "audit"],
            experiment_type="direct_ctmc_validation",
        )
    return run


def _make_ctmc_support_summary(output_dir: str, config_name: str) -> Callable[[], list[Path]]:
    """Create a launcher for the consolidated CTMC support capsule."""
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.verification.ctmc_support_summary",
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=False,
            experiment_type="ctmc_support_summary",
        )
    return run


def _make_check_configs(config_name: str, output_dir: str) -> Callable[[], list[Path]]:
    """Create a launcher for the config testing sanity check."""

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.testing.check_configs",
            config_name=config_name,
            output_dir=None,
            hydra=True,
        )
    return run


def _make_engine_parity(config_name: str, output_dir: str) -> Callable[[], list[Path]]:
    """Create a launcher for the engine parity verification."""
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.verification.engine_parity",
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=True,
            experiment_type="engine_parity",
        )
    return run


def _make_drift_verification(config_name: str, output_dir: str) -> Callable[[], list[Path]]:
    """Create a launcher for the raw drift verification."""
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.verification.drift_verification",
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=True,
            experiment_type="drift",
        )
    return run


def _make_proof_search(config_name: str, output_dir: str) -> Callable[[], list[Path]]:
    """Create a launcher for the reflected UAS proof search."""
    resolved_output_dir = resolve_runner_output_dir(config_name, output_dir)

    def run() -> list[Path]:
        return launch_module(
            module="gibbsq.experiments.verification.reflected_uas_proof_search",
            config_name=config_name,
            output_dir=resolved_output_dir,
            hydra=True,
            experiment_type="proof_search",
        )
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

def build_verification_core_experiments(output_dir: str, config_name: str) -> list[tuple[str, str, Callable]]:
    """Build the publication-facing verification experiments.

    Returns
    -------
    list of (name, hypothesis, callable)
    """
    return [
        (
            "Boundary Equilibrium Verification",
            "H2",
            _make_boundary_equilibrium(output_dir, config_name),
        ),
        (
            "Reflected-ODE Multi-Start Convergence",
            "H1, H2",
            _make_reflected_ode_convergence(output_dir, config_name),
        ),
        (
            "CTMC Generator Boundary-Mismatch Demo",
            "H3",
            _make_boundary_mismatch_demo(output_dir, config_name),
        ),
        (
            "Direct CTMC Validation Capsule",
            "H4",
            _make_direct_ctmc_validation(output_dir, config_name),
        ),
        (
            "Exhaustive Small-Grid Drift Audit",
            "H4",
            _make_exhaustive_drift_audit(output_dir, config_name),
        ),
        (
            "Theorem-Constant Parameter Sweep",
            "H4",
            _make_theorem_constant_sweep(output_dir, config_name),
        ),
        (
            "Consolidated CTMC Support Summary",
            "H3, H4",
            _make_ctmc_support_summary(output_dir, config_name),
        ),
    ]


def build_verification_check_experiments(output_dir: str, config_name: str) -> list[tuple[str, str, Callable]]:
    """Build the support checks that are not paper-facing experiments."""
    return [
        (
            "Configuration Sanity Checks",
            "Pre-flight",
            _make_check_configs(config_name, output_dir),
        ),
        (
            "Engine Parity Check (NumPy vs JAX)",
            "Verification",
            _make_engine_parity(config_name, output_dir),
        ),
        (
            "Theorem-Backed Drift Verification",
            "Verification",
            _make_drift_verification(config_name, output_dir),
        ),
        (
            "Reflected UAS Exploratory Proof Search",
            "Verification",
            _make_proof_search(config_name, output_dir),
        ),
    ]


def build_verification_experiments(output_dir: str, config_name: str) -> list[tuple[str, str, Callable]]:
    """Build the publication-facing verification experiment suite."""
    return build_verification_core_experiments(output_dir, config_name)


def build_verification_full_suite(output_dir: str, config_name: str) -> list[tuple[str, str, Callable]]:
    """Build the legacy combined suite: experiments plus support checks."""
    return [
        *build_verification_core_experiments(output_dir, config_name),
        *build_verification_check_experiments(output_dir, config_name),
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
        default=None,
        help="Output root for experiment capsules. Defaults to the selected config's output_dir.",
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
        help="List experiments without executing them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the verification runner."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    experiments = build_verification_experiments(args.output_dir, args.config_name)

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
