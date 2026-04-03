#!/usr/bin/env python3
"""
Phase 1 bundled runner for the final_experiment campaign.

This phase intentionally excludes the expensive standalone jobs that are run
manually in phase 2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gibbsq.utils.progress import create_progress
from scripts.execution.experiment_runner import default_hydra_overrides_for_experiment
from scripts.execution.reproduction_pipeline import (
    _current_timestamp,
    _format_elapsed,
    _format_pipeline_step,
    _format_timestamp,
    run_experiment,
)


PHASE_1_EXPERIMENTS = (
    "check_configs",
    "reinforce_check",
    "drift",
    "sweep",
    "bc_train",
)

EXPERIMENT_DESCRIPTIONS = {
    "check_configs": "Running Configuration Sanity Checks...",
    "reinforce_check": "Running REINFORCE Gradient validation...",
    "drift": "Running Drift Verification...",
    "sweep": "Running Stability Sweep...",
    "bc_train": "Running Platinum BC Pretraining...",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 bundled runner for the final_experiment campaign")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved experiment commands without executing them")
    parser.add_argument("--start-from", type=str, default=None, help="Experiment alias to start this phase from")
    parser.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Progress bar mode for the bundle and child experiments",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="final_experiment",
        help="Profile config to load (default: final_experiment)",
    )
    args, hydra_args = parser.parse_known_args()

    experiments = list(PHASE_1_EXPERIMENTS)
    if args.start_from:
        if args.start_from not in experiments:
            print(f"Error: Unknown start experiment '{args.start_from}'. Valid options: {', '.join(experiments)}")
            return 1
        experiments = experiments[experiments.index(args.start_from):]

    global_hydra_args = list(hydra_args)
    if not any(arg.startswith("--config-name") for arg in global_hydra_args):
        global_hydra_args.extend(["--config-name", args.config_name])


    start = _current_timestamp()
    start_perf = perf_counter()
    print("=" * 58)
    print("  GibbsQ Final Campaign: Phase 1 Pipeline")
    print("=" * 58)
    print(f"  Progress Mode: {args.progress}")
    print(f"  Pipeline Started At: {_format_timestamp(start)}")
    print(f"  Experiments: {', '.join(experiments)}")

    failed: list[str] = []
    global_hydra_args = list(hydra_args)
    try:
        with create_progress(total=len(experiments), desc="final-phase1", mode=args.progress, unit="experiment") as progress:
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
        print("  Phase 1 Pipeline complete." if not failed else "  Phase 1 Pipeline completed with failures.")
        if failed:
            print(f"  Failed Experiments: {', '.join(failed)}")
        print(f"  Pipeline Ended At: {_format_timestamp(end)}")
        print(f"  Total Pipeline Runtime: {_format_elapsed(total_elapsed)}")
        print("=" * 58)


if __name__ == "__main__":
    sys.exit(main())
