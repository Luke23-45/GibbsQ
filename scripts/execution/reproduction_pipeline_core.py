#!/usr/bin/env python3
"""
Pipeline-safe pre-train execution bundle for the final campaign.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.execution.reproduction_pipeline import _format_pipeline_step, _format_timestamp, _format_elapsed, _current_timestamp, run_experiment
from scripts.execution.experiment_runner import default_hydra_overrides_for_experiment
from gibbsq.utils.progress import create_progress
from scripts.verification.runtime_budgeting import GROUP_A_EXPERIMENTS
from time import perf_counter


EXPERIMENT_DESCRIPTIONS = {
    "check_configs": "Running Configuration Sanity Checks...",
    "reinforce_check": "Running REINFORCE Gradient validation...",
    "drift": "Running Drift Verification...",
    "sweep": "Running Stability Sweep...",
    "stress": "Running Scaling Stress Tests...",
    "bc_train": "Running Platinum BC Pretraining...",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline-safe pre-train execution bundle")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved experiment commands without executing them")
    parser.add_argument("--start-from", type=str, default=None, help="Experiment alias to start this bundle from")
    parser.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Progress bar mode for the bundle and child experiments",
    )
    args, hydra_args = parser.parse_known_args()

    experiments = list(GROUP_A_EXPERIMENTS)
    if args.start_from:
        if args.start_from not in experiments:
            print(f"Error: Unknown start experiment '{args.start_from}'. Valid options: {', '.join(experiments)}")
            return 1
        experiments = experiments[experiments.index(args.start_from):]

    start = _current_timestamp()
    start_perf = perf_counter()
    print("=" * 58)
    print("  GibbsQ Final Campaign: Core Pre-Train Pipeline")
    print("=" * 58)
    print(f"  Progress Mode: {args.progress}")
    print(f"  Pipeline Started At: {_format_timestamp(start)}")
    print(f"  Experiments: {', '.join(experiments)}")

    failed: list[str] = []
    global_hydra_args = list(hydra_args)
    try:
        with create_progress(total=len(experiments), desc="core-pipeline", mode=args.progress, unit="experiment") as progress:
            for idx, experiment in enumerate(experiments, start=1):
                print(f"\n{_format_pipeline_step(EXPERIMENT_DESCRIPTIONS[experiment], idx, len(experiments))}")
                current_args = global_hydra_args + default_hydra_overrides_for_experiment(experiment, global_hydra_args)
                step_start = _current_timestamp()
                step_perf = perf_counter()
                print(f"  -> [{experiment}] Started at {_format_timestamp(step_start)}")
                result = run_experiment(experiment, current_args, dry_run=args.dry_run, progress_mode=args.progress)
                elapsed = perf_counter() - step_perf
                step_end = _current_timestamp()
                print(f"  -> [{experiment}] Ended at {_format_timestamp(step_end)}")
                print(f"  -> [{experiment}] Status: {'completed' if result == 0 else f'failed (exit code {result})'}")
                print(f"  -> [{experiment}] Elapsed: {_format_elapsed(elapsed)}")
                progress.update(1)
                if result != 0:
                    failed.append(experiment)
                    break
        return 1 if failed else 0
    finally:
        end = _current_timestamp()
        total_elapsed = perf_counter() - start_perf
        print("\n" + "=" * 58)
        print("  Core Pipeline complete." if not failed else "  Core Pipeline completed with failures.")
        if failed:
            print(f"  Failed Experiments: {', '.join(failed)}")
        print(f"  Pipeline Ended At: {_format_timestamp(end)}")
        print(f"  Total Pipeline Runtime: {_format_elapsed(total_elapsed)}")
        print("=" * 58)


if __name__ == "__main__":
    sys.exit(main())
