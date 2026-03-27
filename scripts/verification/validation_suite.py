#!/usr/bin/env python3
"""
Unified verification runner for GibbsQ.
Consolidates verify_implementation.ps1 and verify_phase_iv.ps1 into a robust Python CLI.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_cmd(args, dry_run=False):
    """Run a command using the project's run_experiment.py script."""
    script_dir = Path(__file__).parent
    run_script = script_dir.parent / "execution" / "experiment_runner.py"
    
    # Use sys.executable to ensure we use the same environment
    cmd = [sys.executable, str(run_script)] + args
    
    print("\n" + "="*60)
    print(f" COMMAND: {' '.join(cmd)}")
    print("="*60)
    
    if dry_run:
        print("[DRY-RUN] Skipping execution.")
        return
        
    try:
        # We don't set check=True here so we can handle the failure gracefully if needed
        # but let's keep it simple and exit on first failure as PS scripts did.
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL ERROR] Step failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def verify_standard(args):
    """Corresponds to the logic in verify_implementation.ps1."""
    level = args.level
    dry_run = args.dry_run
    
    # Logic mapping from the old PS script
    # Defaults (quick validation)
    settings = {
        "quick":   (500,   1, 0.1, 500,   1, 1),
        "trusted": (5000,  3, 0.2, 5000,  2, 5),
        "full":    (25000, 5, 0.2, 10000, 5, 30)
    }
    
    sim_time, reps, burn_in, policy_time, stress_reps, train_epochs = settings[level]
    
    print(f"\n>>> INITIATING STANDARD VERIFICATION (Level: {level.upper()}) <<<\n")
    
    # Step 0: Pre-flight Configuration Check
    print("[0/6] Running Step 0: Pre-flight configuration check...")
    run_cmd(["check_configs"], dry_run)
    
    # Step 1: Theoretical Drift
    print("\n[1/6] Running Step 1: Theoretical Drift Verification...")
    run_cmd(["drift", "system.num_servers=2", "system.arrival_rate=2.0"], dry_run)
    
    # Step 2: Stability Sweep
    print("\n[2/6] Running Step 2: Stability Sweep...")
    run_cmd([
        "sweep", 
        f"simulation.ssa.sim_time={sim_time}", 
        f"simulation.num_replications={reps}", 
        f"simulation.burn_in_fraction={burn_in}"
    ], dry_run)
    
    # Step 3: Policy Comparison
    print("\n[3/6] Running Step 3: Policy Comparison (Heterogeneous)...")
    run_cmd([
        "policy", 
        "system.num_servers=4", 
        "system.service_rates=[1.0, 2.0, 5.0, 10.0]", 
        f"simulation.ssa.sim_time={policy_time}", 
        f"simulation.num_replications={reps}"
    ], dry_run)
    
    # Step 4: Stress Test
    print("\n[4/6] Running Step 4: Stress Test...")
    run_cmd([
        "stress", 
        f"simulation.num_replications={stress_reps}", 
        "jax.enabled=true"
    ], dry_run)
    
    # Step 5: Training (Phase 2: Neural Learning Pipeline)
    print("\n[5/6] Running Step 5: REINFORCE Training...")
    run_cmd(["reinforce_train", f"train_epochs={train_epochs}"], dry_run)

def verify_phase_iv(args):
    """Corresponds to the logic in verify_phase_iv.ps1."""
    config = args.config_name
    dry_run = args.dry_run
    # Re-packing any remaining hydra args
    hydra_args = args.hydra_args
    
    print(f"\n>>> INITIATING PHASE IV CORRECTIVE TRACK VERIFICATION (Config: {config}) <<<\n")
    
    common = [f"--config-name={config}"] + hydra_args
    
    # Pre-flight: Configuration Check
    print("[0/4] Pre-flight Check: Validating configuration...")
    run_cmd(["check_configs"] + common, dry_run)
    
    # Track 5: Gradient Estimator Validation
    print("\n[1/4] Track 5: Validating Gradient Estimator...")
    run_cmd(["reinforce_check"] + common, dry_run)
    
    # Track 1: REINFORCE SSA Training
    print("\n[2/4] Track 1: Training REINFORCE Agent (True SSA)...")
    run_cmd(["reinforce_train"] + common, dry_run)
    
    # Track 3: Platinum BC Pretraining
    print("\n[3/4] Track 2/3: Platinum Behavior Cloning Pretraining (Robust)...")
    run_cmd(["bc_train"] + common, dry_run)
    
    # Track 4: Platinum Benchmark
    print("\n[4/4] Track 4: Running Platinum Parity Benchmark (Corrected Grid)...")
    run_cmd(["policy"] + common, dry_run)

def verify_full_paper(args):
    """Executes the entire research pipeline from Phases 0-3."""
    config = args.config_name
    dry_run = args.dry_run
    hydra_args = args.hydra_args
    
    print(f"\n>>> INITIATING FULL PAPER PIPELINE (Config: {config}) <<<\n")
    
    common = [f"--config-name={config}"] + hydra_args
    
    # PHASE 0: Pre-flight & Validation
    print("--- [PHASE 0] Pre-flight & Validation ---")
    run_cmd(["check_configs"] + common, dry_run)
    run_cmd(["reinforce_check"] + common, dry_run)
    
    # PHASE 1: Baselines & Metrics
    print("\n--- [PHASE 1] Baselines & Foundational Metrics ---")
    run_cmd(["drift"] + common, dry_run)
    run_cmd(["sweep"] + common, dry_run)
    run_cmd(["stress"] + common, dry_run)
    run_cmd(["policy"] + common, dry_run)
    
    # PHASE 2: Training
    print("\n--- [PHASE 2] Neural Learning Pipeline ---")
    run_cmd(["bc_train"] + common, dry_run)
    run_cmd(["reinforce_train"] + common, dry_run)
    
    # PHASE 3: Analysis
    print("\n--- [PHASE 3] Deep Generational Analysis & Ablation ---")
    run_cmd(["stats"] + common, dry_run)
    run_cmd(["generalize"] + common, dry_run)
    run_cmd(["ablation"] + common, dry_run)
    run_cmd(["critical"] + common, dry_run)

def main():
    # Common parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--dry-run", action="store_true", help="Display commands without executing them")

    parser = argparse.ArgumentParser(
        description="Unified Verification Suite for GibbsQ Research Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Target verification workflow")
    
    # 'standard' subcommand
    parser_std = subparsers.add_parser(
        "standard", 
        parents=[parent_parser],
        help="Implementation verification (Stability, Drift, Stress, Training)"
    )
    parser_std.add_argument("--level", choices=["quick", "trusted", "full"], default="quick", 
                            help="Rigor level determining simulation time and metrics precision")
    
    # 'phase_iv' subcommand
    parser_iv = subparsers.add_parser(
        "phase_iv", 
        parents=[parent_parser],
        help="Phase IV Corrective Track verification (REINFORCE + SSA)"
    )
    parser_iv.add_argument("--config-name", default="small", help="Hydra configuration profile to use")
    parser_iv.add_argument("hydra_args", nargs="*", help="Arbitrary Hydra configuration overrides")
    
    # 'full_paper' subcommand
    parser_full = subparsers.add_parser(
        "full_paper",
        parents=[parent_parser],
        help="Execute the full end-to-end paper pipeline (Phases 0-3)"
    )
    parser_full.add_argument("--config-name", default="small", help="Hydra configuration profile to use")
    parser_full.add_argument("hydra_args", nargs="*", help="Arbitrary Hydra configuration overrides")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    try:
        if args.command == "standard":
            verify_standard(args)
        elif args.command == "phase_iv":
            verify_phase_iv(args)
        elif args.command == "full_paper":
            verify_full_paper(args)
            
        print("\n" + "="*60)
        print("  VERIFICATION COMPLETE!")
        print("="*60)
        if not args.dry_run:
            print("  Check '/outputs' for detailed analysis results.")
    
    except KeyboardInterrupt:
        print("\n[Interrupted] Verification canceled by user.")
        sys.exit(130)

if __name__ == "__main__":
    main()



