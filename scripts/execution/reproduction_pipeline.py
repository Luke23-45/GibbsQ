#!/usr/bin/env python3
"""
Cross-platform master execution pipeline for GibbsQ & N-GibbsQ research paper.
Replaces both run_paper_experiments.ps1 and run_paper_experiments.sh.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_experiment(experiment, hydra_args=None):
    """Run a single experiment using the unified experiment_runner.py script."""
    if hydra_args is None:
        hydra_args = []
    
    script_dir = Path(__file__).parent
    run_script = script_dir / "experiment_runner.py"
    
    cmd = ["python", str(run_script), experiment] + hydra_args
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment '{experiment}': {e}")
        return e.returncode

def main():
    """Run the complete paper experiment pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cross-platform master execution pipeline for GibbsQ research paper"
    )
    parser.add_argument("--start-from", type=str, default=None, help="Experiment alias to start the pipeline from (e.g. 'drift', 'policy')")
    
    args, hydra_args = parser.parse_known_args()
    
    print("=" * 58)
    print("  GibbsQ Research Paper: Final Execution Pipeline")
    print("=" * 58)
    print("\n[Initiating Pipeline...]\n")
    
    experiments = [
        # Phase 0: Gradient Estimator Validation
        ("check_configs", "[Pre-Flight] Running Configuration Sanity Checks..."),
        ("reinforce_check", "[0/10] Running REINFORCE Gradient validation (Track 5)..."),
        
        # Phase 1: Foundational Analytical Metrics & Verification  
        ("drift", "[1/10] Running Drift Verification (Phase 1a)..."),
        ("sweep", "[2/10] Running Stability Sweep (Phase 1b)..."),
        ("stress", "[3/10] Running Scaling Stress Tests (Phase 1c)..."),
        
        # Phase 2: N-GibbsQ Neural Learning Pipeline
        ("bc_train", "[4/10] Running Platinum BC Pretraining (Phase 2a)..."),
        ("reinforce_train", "[5/10] Running REINFORCE SSA Training (Phase 2b)..."),
        
        # Phase 3: Evaluation benchmarks (requires weights from Phase 2)
        ("policy", "[6/10] Running Corrected Policy Evaluation Benchmark (Phase 3a)..."),
        ("stats", "[7/10] Running Statistical Verification Analysis (Phase 3b)..."),
        
        # Phase 4: Generational Analysis & Ablation
        ("generalize", "[8/10] Running Generalization Stress Heatmaps..."),
        ("critical", "[9/10] Running Critical Load Boundary Analysis..."),
        ("ablation", "[10/10] Running SSA Component Ablation..."),
    ]
    
    # Process --start-from logic
    if args.start_from:
        # Find index of start experiment
        found_index = -1
        for i, (exp_alias, _) in enumerate(experiments):
            if exp_alias == args.start_from:
                found_index = i
                break
        
        if found_index == -1:
            valid_aliases = [exp for exp, _ in experiments]
            print(f"Error: Unknown start experiment '{args.start_from}'")
            print(f"Valid options are: {', '.join(valid_aliases)}")
            return 1
        
        experiments = experiments[found_index:]
        print(f"  [Resuming pipeline from step: {args.start_from}]\n")

    
    # Special arguments for specific experiments
    special_args = {
        "sweep": ["+experiment=stability_sweep"],
        "stress": ["++jax.enabled=True"],
    }
    
    failed_experiments = []
    
    global_hydra_args = list(hydra_args)
    
    for i, (experiment, description) in enumerate(experiments):
        if description:
            print(f"\n{description}")
        
        current_hydra_args = global_hydra_args + special_args.get(experiment, [])
        
        print(f"Running: {experiment}")
        result = run_experiment(experiment, current_hydra_args)
        
        if result != 0:
            print(f"Experiment '{experiment}' failed with exit code {result}")
            failed_experiments.append(experiment)
    
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
