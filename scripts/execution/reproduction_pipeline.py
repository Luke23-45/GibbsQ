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
    parser.add_argument("hydra_args", nargs="*", help="Hydra configuration overrides")
    
    args = parser.parse_args()
    
    print("=" * 58)
    print("  GibbsQ Research Paper: Final Execution Pipeline")
    print("=" * 58)
    print("\n[Initiating Pipeline...]\n")
    
    experiments = [
        # Phase 0: Gradient Estimator Validation
        ("reinforce_check", "[0/10] Running REINFORCE Gradient validation (Track 5)..."),
        
        # Phase 1: Foundational Analytical Metrics & Verification  
        ("drift", "[1/11] Running Drift Verification (Phase 1a)..."),
        # Note: "fidelity" experiment removed as it's deprecated
        ("corrected_policy", "[4/10] Running Corrected Policy Evaluation Benchmark (Track 4)..."),
        ("sweep", "[5/10] Running Stability Sweep..."),
        ("stress", "[6/10] Running Scaling Stress Tests..."),
        
        # Phase 2: N-GibbsQ Neural Learning Pipeline
        ("reinforce_train", "[7/10] Running REINFORCE SSA Training (Track 1)..."),
        ("dr_train", "[8/10] Running Domain Randomization Training (Track 3)..."),
        
        # Phase 3: Generational Analysis & Ablation
        ("stats", "[10/10] Running Deep Component Ablation & Generational Generalization..."),
        ("generalize", ""),
        ("critical", ""),
        ("ablation", ""),
    ]
    
    # Special arguments for specific experiments
    special_args = {
        "sweep": ["+experiment=stability_sweep"],
        "stress": ["++jax.enabled=True"],
    }
    
    failed_experiments = []
    
    for i, (experiment, description) in enumerate(experiments):
        if description:
            print(f"\n{description}")
        
        hydra_args = args.hydra_args + special_args.get(experiment, [])
        
        print(f"Running: {experiment}")
        result = run_experiment(experiment, hydra_args)
        
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
