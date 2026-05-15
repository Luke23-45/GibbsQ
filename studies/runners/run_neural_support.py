#!/usr/bin/env python3
"""
Run the supporting neural applicability experiments.

These experiments support the H7 applicability layer. They are not
theorem evidence and are intentionally kept separate from the z2
publication-facing verification pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from gibbsq.qroute.utils.model_io import resolve_model_pointer
from studies.runners.common import (
    PROJECT_ROOT,
    launch_module,
    resolve_config_output_dir,
    resolve_runner_output_dir,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "final_experiment"
DEFAULT_REPORT_DIR = "outputs/reports"


@dataclass
class ExperimentResult:
    name: str
    hypothesis: str
    module: str
    status: str = "NOT_RUN"
    elapsed_seconds: float = 0.0
    output_files: list[str] = field(default_factory=list)
    error_message: str | None = None
    traceback: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "module": self.module,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "output_files": self.output_files,
            "error_message": self.error_message,
            "notes": list(self.notes),
        }


def _experiment_type_for_module(module: str) -> str:
    mapping = {
        "gibbsq.experiments.evaluation.baselines_comparison": "policy",
        "gibbsq.experiments.evaluation.n_gibbsq_evals.stats_bench": "stats",
        "gibbsq.experiments.evaluation.n_gibbsq_evals.gen_sweep": "generalize",
        "gibbsq.experiments.evaluation.n_gibbsq_evals.critical_load": "critical",
        "gibbsq.experiments.evaluation.n_gibbsq_evals.ablation_ssa": "ablation",
    }
    return mapping[module]


def _run_module(
    *,
    name: str,
    hypothesis: str,
    module: str,
    config_name: str,
    output_dir: str | None,
    extra_overrides: Sequence[str] = (),
) -> ExperimentResult:
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
            extra_args=extra_overrides,
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


def build_neural_experiments(config_name: str) -> list[tuple[str, str, str, tuple[str, ...]]]:
    return [
        (
            "Corrected Policy Comparison",
            "H7",
            "gibbsq.experiments.evaluation.baselines_comparison",
            (),
        ),
        (
            "Neural Statistical Benchmark",
            "H7",
            "gibbsq.experiments.evaluation.n_gibbsq_evals.stats_bench",
            (),
        ),
        (
            "Neural Generalization Sweep",
            "H7",
            "gibbsq.experiments.evaluation.n_gibbsq_evals.gen_sweep",
            (),
        ),
        (
            "Neural Critical-Load Study",
            "H7",
            "gibbsq.experiments.evaluation.n_gibbsq_evals.critical_load",
            (),
        ),
        (
            "Neural SSA Ablation",
            "H7",
            "gibbsq.experiments.evaluation.n_gibbsq_evals.ablation_ssa",
            (),
        ),
    ]


def _resolve_neural_pointer(output_root: Path, *, allow_bc: bool) -> Path:
    return resolve_model_pointer(
        PROJECT_ROOT,
        output_root,
        allow_bc=allow_bc,
        allow_legacy=False,
    )


def _run_bc_bootstrap(config_name: str, output_root: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gibbsq.experiments.training.pretrain_bc",
        "--config-name",
        config_name,
        f"++active_profile={config_name}",
        f"++output_dir={output_root}",
    ]
    log.info("Launching BC bootstrap: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _preflight_ensure_neural_weights(config_name: str, output_root: Path) -> tuple[Path, str]:
    try:
        return _resolve_neural_pointer(output_root, allow_bc=False), "reinforce"
    except FileNotFoundError:
        try:
            return _resolve_neural_pointer(output_root, allow_bc=True), "bc"
        except FileNotFoundError:
            log.warning(
                "No public neural pointer found in %s. Running BC bootstrap before neural support.",
                output_root,
            )
            _run_bc_bootstrap(config_name, output_root)
            try:
                return _resolve_neural_pointer(output_root, allow_bc=True), "bc_bootstrap"
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "BC bootstrap completed but no usable neural pointer was created under "
                    f"{output_root}. Expected latest_reinforce_weights.txt or latest_bc_weights.txt."
                ) from exc


def _write_report(results: list[ExperimentResult], report_dir: str | Path, total_elapsed: float) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    json_report = {
        "pipeline": "neural_support",
        "timestamp": timestamp,
        "overall_status": overall,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "summary": {"total": len(results), "pass": n_pass, "fail": n_fail},
        "experiments": [r.to_dict() for r in results],
    }
    json_path = report_dir / f"neural_support_report_{timestamp}.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

    lines = [
        f"# Neural Support Pipeline Report - {timestamp}",
        "",
        "These runs support the H7 applicability claim only.",
        "",
        f"Overall: {overall} | Pass: {n_pass} | Fail: {n_fail} | Wall time: {total_elapsed:.1f}s",
        "",
        "| # | Experiment | Module | Status | Time (s) | Output Files |",
        "|---|-----------|--------|--------|----------|-------------|",
    ]
    for idx, r in enumerate(results, start=1):
        files_str = ", ".join(Path(f).name for f in r.output_files) or "-"
        lines.append(
            f"| {idx} | {r.name} | {r.module} | {r.status} | {r.elapsed_seconds:.1f} | {files_str} |"
        )
    if n_fail > 0:
        lines.extend(["", "## Failures", ""])
        for r in results:
            if r.status == "FAIL":
                lines.append(f"### {r.name}")
                lines.append(f"```\n{r.error_message}\n```")
                lines.append("")
    md_path = report_dir / f"neural_support_report_{timestamp}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][neural] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the supporting neural applicability studies. "
            "These are not part of the main theorem-backed z2 pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root for experiment capsules. Defaults to the selected config's output_dir.",
    )
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    experiments = build_neural_experiments(args.config_name)
    if args.dry_run:
        log.info("DRY RUN - %d neural-support experiments would be executed:", len(experiments))
        for name, hyp, module, _ in experiments:
            log.info("  - %s (%s) -> %s", name, hyp, module)
        return 0

    resolved_output_dir = resolve_runner_output_dir(args.config_name, args.output_dir)

    try:
        model_path, weight_source = _preflight_ensure_neural_weights(args.config_name, resolved_output_dir)
        log.info("Neural preflight OK - using %s weights from %s", weight_source, model_path)
    except Exception as exc:
        log.error("Neural preflight failed: %s", exc)
        log.error(
            "Ensure the configured output directory is writable and BC training can complete under %s.",
            resolve_config_output_dir(args.config_name),
        )
        return 1

    results: list[ExperimentResult] = []
    t0 = time.perf_counter()
    for name, hyp, module, extra_overrides in experiments:
        results.append(
            _run_module(
                name=name,
                hypothesis=hyp,
                module=module,
                config_name=args.config_name,
                output_dir=str(resolved_output_dir),
                extra_overrides=extra_overrides,
            )
        )
    total_elapsed = time.perf_counter() - t0
    report_path = _write_report(results, args.report_dir, total_elapsed)
    log.info("Report: %s", report_path)
    return 0 if all(r.status == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
