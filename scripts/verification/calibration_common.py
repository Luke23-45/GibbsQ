from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verification.runtime_budgeting import (
    append_jsonl,
    apply_feature_overrides,
    default_calibration_matrix,
    experiment_runtime_features,
    load_stage_profile,
    now_iso,
    project_relative,
)


def _build_parser(experiment_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Fixed compute-control calibration runner for {experiment_name}",
    )
    parser.add_argument("--config-name", default="small", help="Base profile to calibrate against")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "runtime_calibration" / f"{experiment_name}.jsonl",
        help="JSONL file that will receive one row per calibration run",
    )
    parser.add_argument("--limit-runs", type=int, default=None, help="Cap the number of calibration rows executed")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands without executing them")
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override appended to every calibration command",
    )
    return parser


def _run_one(
    experiment_name: str,
    profile_name: str,
    feature_override: dict[str, float],
    output_path: Path,
    extra_overrides: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    before = perf_counter()
    output_root = PROJECT_ROOT / "outputs"
    stage_before = load_stage_profile(output_root, profile_name, experiment_name)
    overrides = apply_feature_overrides(experiment_name, feature_override) + list(extra_overrides)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "execution" / "experiment_runner.py"),
        "--config-name",
        profile_name,
        "--progress",
        "off",
        experiment_name,
        *overrides,
    ]
    record: dict[str, Any] = {
        "timestamp": now_iso(),
        "experiment": experiment_name,
        "config_name": profile_name,
        "feature_override": feature_override,
        "hydra_overrides": overrides,
        "command": cmd,
        "dry_run": dry_run,
    }
    if dry_run:
        record["status"] = "planned"
        append_jsonl(output_path, record)
        return record

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = perf_counter() - before
    stage_after = load_stage_profile(output_root, profile_name, experiment_name)
    resolved_features = experiment_runtime_features(profile_name, experiment_name)
    record.update(
        {
            "status": "completed" if result.returncode == 0 else "failed",
            "return_code": int(result.returncode),
            "wall_time_seconds": elapsed,
            "resolved_features": resolved_features,
            "stage_profile_before": stage_before,
            "stage_profile_after": stage_after,
        }
    )
    append_jsonl(output_path, record)
    return record


def run_calibration(experiment_name: str) -> int:
    parser = _build_parser(experiment_name)
    args = parser.parse_args()

    rows = default_calibration_matrix(args.config_name, experiment_name)
    if args.limit_runs is not None:
        rows = rows[: args.limit_runs]

    print("=" * 58)
    print(f" Calibration Runner: {experiment_name}")
    print("=" * 58)
    print(f" Base Profile: {args.config_name}")
    print(f" Output: {project_relative(args.output)}")
    print(f" Planned Rows: {len(rows)}")
    print(f" Dry Run: {args.dry_run}")

    failures = 0
    for idx, row in enumerate(rows, start=1):
        print(f"\n[{idx}/{len(rows)}] override={json.dumps(row, sort_keys=True)}")
        result = _run_one(
            experiment_name=experiment_name,
            profile_name=args.config_name,
            feature_override=row,
            output_path=args.output,
            extra_overrides=args.extra_override,
            dry_run=args.dry_run,
        )
        print(f"  -> status={result['status']}")
        if result["status"] == "failed":
            failures += 1
            break
    return 1 if failures else 0
