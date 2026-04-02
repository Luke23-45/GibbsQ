#!/usr/bin/env python3
"""
Unified verification runner for GibbsQ.
Consolidates verify_implementation.ps1 and verify_phase_iv.ps1 into one Python CLI.
"""

import sys
import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter


def _current_timestamp() -> datetime:
    """Return the current local time with timezone information attached."""
    return datetime.now().astimezone()


def _format_timestamp(timestamp: datetime) -> str:
    """Render timestamps in an unambiguous local format."""
    return timestamp.isoformat(sep=" ", timespec="seconds")


def _format_elapsed(elapsed_seconds: float) -> str:
    return f"{elapsed_seconds:.3f}s"


def _format_validation_step(step_idx: int, total_steps: int, label: str) -> str:
    return f"[{step_idx}/{total_steps}] {label}"


def run_cmd(args, dry_run=False):
    script_dir = Path(__file__).parent
    run_script = script_dir.parent / "execution" / "experiment_runner.py"
    
    cmd = [sys.executable, str(run_script)] + args
    step_name = args[0] if args else "unknown"
    step_start = _current_timestamp()
    step_start_perf = perf_counter()
    step_status = "unknown"
    
    print("\n" + "="*60)
    print(f" COMMAND: {' '.join(cmd)}")
    print(f" STARTED: {_format_timestamp(step_start)}")
    print("="*60)
    
    try:
        if dry_run:
            print("[DRY-RUN] Skipping execution.")
            step_status = "dry-run"
        else:
            subprocess.run(cmd, check=True)
            step_status = "completed"
    except subprocess.CalledProcessError as e:
        step_status = f"failed (exit code {e.returncode})"
        print(f"\n[CRITICAL ERROR] Step failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        step_status = "interrupted by user"
        raise
    finally:
        step_end = _current_timestamp()
        elapsed_seconds = perf_counter() - step_start_perf
        print(f"[STEP COMPLETE] {step_name}")
        print(f"  Status: {step_status}")
        print(f"  Ended At: {_format_timestamp(step_end)}")
        print(f"  Elapsed: {_format_elapsed(elapsed_seconds)}")

def verify_standard(args):
    """Corresponds to the logic in verify_implementation.ps1."""
    level = args.level
    dry_run = args.dry_run
    
    settings = {
        "quick":   (500,   1, 0.1, 500,   1, 1),
        "trusted": (5000,  3, 0.2, 5000,  2, 5),
        "full":    (25000, 5, 0.2, 10000, 5, 30)
    }
    
    sim_time, reps, burn_in, policy_time, stress_reps, train_epochs = settings[level]
    
    print(f"\n>>> INITIATING STANDARD VERIFICATION (Level: {level.upper()}) <<<\n")
    
    total_steps = 6

    print(_format_validation_step(1, total_steps, "Running pre-flight configuration check..."))
    run_cmd(["check_configs"], dry_run)
    
    print(f"\n{_format_validation_step(2, total_steps, 'Running theoretical drift verification...')}")
    run_cmd(["drift", "system.num_servers=2", "system.arrival_rate=2.0"], dry_run)
    
    print(f"\n{_format_validation_step(3, total_steps, 'Running stability sweep...')}")
    run_cmd([
        "sweep", 
        f"simulation.ssa.sim_time={sim_time}", 
        f"simulation.num_replications={reps}", 
        f"simulation.burn_in_fraction={burn_in}"
    ], dry_run)
    
    print(f"\n{_format_validation_step(4, total_steps, 'Running policy comparison (heterogeneous)...')}")
    run_cmd([
        "policy", 
        "system.num_servers=4", 
        "system.service_rates=[1.0, 2.0, 5.0, 10.0]", 
        f"simulation.ssa.sim_time={policy_time}", 
        f"simulation.num_replications={reps}"
    ], dry_run)
    
    print(f"\n{_format_validation_step(5, total_steps, 'Running stress test...')}")
    run_cmd([
        "stress", 
        f"simulation.num_replications={stress_reps}", 
        "jax.enabled=true"
    ], dry_run)
    
    print(f"\n{_format_validation_step(6, total_steps, 'Running REINFORCE training...')}")
    run_cmd(["reinforce_train", f"train_epochs={train_epochs}"], dry_run)

def verify_phase_iv(args):
    """Corresponds to the logic in verify_phase_iv.ps1."""
    config = args.config_name
    dry_run = args.dry_run
    hydra_args = args.hydra_args
    
    print(f"\n>>> INITIATING PHASE IV CORRECTIVE TRACK VERIFICATION (Config: {config}) <<<\n")
    
    common = [f"--config-name={config}"] + hydra_args
    
    total_steps = 5

    print(_format_validation_step(1, total_steps, "Pre-flight check: validating configuration..."))
    run_cmd(["check_configs"] + common, dry_run)
    
    print(f"\n{_format_validation_step(2, total_steps, 'Validating REINFORCE gradient estimator...')}")
    run_cmd(["reinforce_check"] + common, dry_run)
    
    print(f"\n{_format_validation_step(3, total_steps, 'Training REINFORCE agent (true SSA)...')}")
    run_cmd(["reinforce_train"] + common, dry_run)
    
    print(f"\n{_format_validation_step(4, total_steps, 'Running platinum behavior cloning pretraining...')}")
    run_cmd(["bc_train"] + common, dry_run)
    
    print(f"\n{_format_validation_step(5, total_steps, 'Running platinum parity benchmark...')}")
    run_cmd(["policy"] + common, dry_run)

def verify_full_paper(args):
    config = args.config_name
    dry_run = args.dry_run
    hydra_args = args.hydra_args
    
    print(f"\n>>> INITIATING FULL PAPER PIPELINE (Config: {config}) <<<\n")
    
    common = [f"--config-name={config}"] + hydra_args
    
    print("--- [PHASE 0] Pre-flight & Validation ---")
    run_cmd(["check_configs"] + common, dry_run)
    run_cmd(["reinforce_check"] + common, dry_run)
    
    print("\n--- [PHASE 1] Baselines & Foundational Metrics ---")
    run_cmd(["drift"] + common, dry_run)
    run_cmd(["sweep"] + common, dry_run)
    run_cmd(["stress"] + common, dry_run)
    run_cmd(["policy"] + common, dry_run)
    
    print("\n--- [PHASE 2] Neural Learning Pipeline ---")
    run_cmd(["bc_train"] + common, dry_run)
    run_cmd(["reinforce_train"] + common, dry_run)
    
    print("\n--- [PHASE 3] Deep Generational Analysis & Ablation ---")
    run_cmd(["stats"] + common, dry_run)
    run_cmd(["generalize"] + common, dry_run)
    run_cmd(["ablation"] + common, dry_run)
    run_cmd(["critical"] + common, dry_run)

def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--dry-run", action="store_true", help="Display commands without executing them")

    parser = argparse.ArgumentParser(
        description="Unified Verification Suite for GibbsQ Research Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Target verification workflow")
    
    parser_std = subparsers.add_parser(
        "standard", 
        parents=[parent_parser],
        help="Implementation verification (Stability, Drift, Stress, Training)"
    )
    parser_std.add_argument("--level", choices=["quick", "trusted", "full"], default="quick", 
                            help="Rigor level determining simulation time and metrics precision")
    
    parser_iv = subparsers.add_parser(
        "phase_iv", 
        parents=[parent_parser],
        help="Phase IV Corrective Track verification (REINFORCE + SSA)"
    )
    parser_iv.add_argument("--config-name", default="small", help="Hydra configuration profile to use")
    parser_iv.add_argument("hydra_args", nargs="*", help="Arbitrary Hydra configuration overrides")
    
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
    
    suite_status = "unknown"
    suite_start = _current_timestamp()
    suite_start_perf = perf_counter()

    try:
        print(f"\nValidation Suite Started At: {_format_timestamp(suite_start)}")
        
        if args.command == "standard":
            verify_standard(args)
        elif args.command == "phase_iv":
            verify_phase_iv(args)
        elif args.command == "full_paper":
            verify_full_paper(args)
            
        suite_status = "completed"
        print("\n" + "="*60)
        print("  VERIFICATION COMPLETE!")
    except KeyboardInterrupt:
        suite_status = "interrupted by user"
        print("\n[Interrupted] Verification canceled by user.")
        sys.exit(130)
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        suite_status = f"failed (exit code {exit_code})"
        raise
    finally:
        suite_end = _current_timestamp()
        print("\n" + "="*60)
        print(f"  Validation Status: {suite_status}")
        print(f"  Ended At: {_format_timestamp(suite_end)}")
        print(f"  Total Elapsed Time: {_format_elapsed(perf_counter() - suite_start_perf)}")
        print("="*60)
        if suite_status == "completed" and not args.dry_run:
            print("  Check '/outputs' for detailed analysis results.")

if __name__ == "__main__":
    main()
