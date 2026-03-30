#!/usr/bin/env python3
"""
Cross-platform master execution pipeline for GibbsQ & N-GibbsQ research paper.
Replaces both run_paper_experiments.ps1 and run_paper_experiments.sh.
"""

import sys
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.execution.experiment_runner import default_hydra_overrides_for_experiment
from gibbsq.utils.progress import create_progress


def run_experiment(experiment, hydra_args=None, dry_run: bool = False, progress_mode: str = "auto"):
    if hydra_args is None:
        hydra_args = []
    
    script_dir = Path(__file__).parent
    run_script = script_dir / "experiment_runner.py"
    
    cmd = [sys.executable, str(run_script), "--progress", progress_mode, experiment] + hydra_args
    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
        return 0
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment '{experiment}': {e}")
        return e.returncode


def _format_pipeline_step(label: str, step_idx: int | None, total_steps: int) -> str:
    if step_idx is None:
        return f"[Pre-Flight] {label}"
    return f"[{step_idx}/{total_steps}] {label}"

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cross-platform master execution pipeline for GibbsQ research paper"
    )
    parser.add_argument("--start-from", type=str, default=None, help="Experiment alias to start the pipeline from (e.g. 'drift', 'policy')")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved experiment commands without executing them")
    parser.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Progress bar mode for the pipeline and child experiments (default: auto)",
    )
    
    args, hydra_args = parser.parse_known_args()
    
    print("=" * 58)
    print("  GibbsQ Research Paper: Final Execution Pipeline")
    print("=" * 58)
    print(f"  Progress Mode: {args.progress}")
    print("\n[Initiating Pipeline...]\n")
    
    experiments = [
        # Phase 0: Gradient Estimator Validation
        ("check_configs", "Running Configuration Sanity Checks...", None),
        ("reinforce_check", "Running REINFORCE Gradient validation...", 1),

        # Phase 1: Foundational Analytical Metrics & Verification
        ("drift", "Running Drift Verification (Phase 1a)...", 2),
        ("sweep", "Running Stability Sweep (Phase 1b)...", 3),
        ("stress", "Running Scaling Stress Tests (Phase 1c)...", 4),

        # Phase 2: N-GibbsQ Neural Learning Pipeline
        ("bc_train", "Running Platinum BC Pretraining (Phase 2a)...", 5),
        ("reinforce_train", "Running REINFORCE SSA Training (Phase 2b)...", 6),

        # Phase 3: Evaluation benchmarks (requires weights from Phase 2)
        ("policy", "Running Corrected Policy Evaluation Benchmark (Phase 3a)...", 7),
        ("stats", "Running Statistical Verification Analysis (Phase 3b)...", 8),

        # Phase 4: Generational Analysis & Ablation
        ("generalize", "Running Generalization Stress Heatmaps...", 9),
        ("critical", "Running Critical Load Boundary Analysis...", 10),
        ("ablation", "Running SSA Component Ablation...", 11),
    ]
    total_numbered_steps = max(step_idx or 0 for _, _, step_idx in experiments)
    
    # Process --start-from logic
    if args.start_from:
        # Find index of start experiment
        found_index = -1
        for i, (exp_alias, _, _) in enumerate(experiments):
            if exp_alias == args.start_from:
                found_index = i
                break
        
        if found_index == -1:
            valid_aliases = [exp for exp, _, _ in experiments]
            print(f"Error: Unknown start experiment '{args.start_from}'")
            print(f"Valid options are: {', '.join(valid_aliases)}")
            return 1
        
        experiments = experiments[found_index:]
        print(f"  [Resuming pipeline from step: {args.start_from}]\n")

    
    failed_experiments = []
    
    global_hydra_args = list(hydra_args)
    
    with create_progress(
        total=len(experiments),
        desc="pipeline",
        mode=args.progress,
        unit="experiment",
    ) as pipeline_bar:
        for index, (experiment, description, step_idx) in enumerate(experiments, start=1):
            rendered_description = _format_pipeline_step(description, step_idx, total_numbered_steps)
            if description:
                print(f"\n{rendered_description}")

            pipeline_bar.set_description(f"{index}/{len(experiments)} {experiment}")
            pipeline_bar.set_postfix({"alias": experiment}, refresh=False)

            current_hydra_args = global_hydra_args + default_hydra_overrides_for_experiment(
                experiment,
                global_hydra_args,
            )

            print(f"[handoff] Launching {experiment} ({index}/{len(experiments)})")
            result = run_experiment(
                experiment,
                current_hydra_args,
                dry_run=args.dry_run,
                progress_mode=args.progress,
            )
            pipeline_bar.update(1)

            if result != 0:
                print(f"Experiment '{experiment}' failed with exit code {result}")
                failed_experiments.append(experiment)
                print("Stopping pipeline after the first failure to avoid running dependent phases on invalid artifacts.")
                break
    
    print("\n" + "=" * 58)
    if failed_experiments:
        print(f"  Pipeline completed with {len(failed_experiments)} failed experiments:")
        for exp in failed_experiments:
            print(f"    - {exp}")
    else:
        print("  Pipeline fully complete.")
    print("  Review '/outputs/' for your plots and logs.")
    print("=" * 58)
    
    return 1 if failed_experiments else 0

if __name__ == "__main__":
    sys.exit(main())
