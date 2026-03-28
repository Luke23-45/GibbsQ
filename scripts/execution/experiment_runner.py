#!/usr/bin/env python3
"""
Cross-platform execution script for GibbsQ experiments.
Replaces both run_experiment.ps1 and run_experiment.sh with a single Python script.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gibbsq.utils.progress import configure_progress_mode

# Experiment mappings
EXPERIMENTS = {
    # Phase 0: Validation & Pre-flight Sanity
    "check_configs": "experiments.testing.check_configs",
    "reinforce_check": "experiments.testing.reinforce_gradient_check",

    # Phase 1: Foundational Metrics & Baselines
    "drift": "experiments.verification.drift_verification",
    "sweep": "experiments.sweeps.stability_sweep",
    "stress": "experiments.testing.stress_test",
    "policy": "experiments.evaluation.baselines_comparison",

    # Phase 2: Neural Learning Pipeline (Training)
    "bc_train": "experiments.training.pretrain_bc",
    "reinforce_train": "experiments.training.train_reinforce",

    # Phase 3: Deep Generational Analysis & Ablation
    "stats": "experiments.evaluation.n_gibbsq_evals.stats_bench",
    "generalize": "experiments.evaluation.n_gibbsq_evals.gen_sweep",
    "ablation": "experiments.evaluation.n_gibbsq_evals.ablation_ssa",
    "critical": "experiments.evaluation.n_gibbsq_evals.critical_load",
}


def _has_override(args: list[str], key: str) -> bool:
    """Return True when Hydra overrides already pin ``key``."""
    key_prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    return any(arg.startswith(key_prefixes) for arg in args)


def default_hydra_overrides_for_experiment(
    experiment: str,
    existing_overrides: list[str] | None = None,
) -> list[str]:
    """Return publication-safe default Hydra overrides for public experiments."""
    existing = list(existing_overrides or [])
    defaults: list[str] = []

    if experiment in {"sweep", "stats", "policy"} and not _has_override(existing, "experiment"):
        experiment_profiles = {
            "sweep": "stability_sweep",
            "stats": "stats_bench",
            "policy": "policy_comparison",
        }
        defaults.append(f"+experiment={experiment_profiles[experiment]}")

    if experiment == "stress" and not _has_override(existing, "jax.enabled"):
        defaults.append("++jax.enabled=True")

    return defaults

def print_usage():
    """Print usage information."""
    print("Usage: python experiment_runner.py [--progress auto|on|off] <experiment> [hydra_args...]")
    print("")
    print("Available experiments:")
    for exp in EXPERIMENTS:
        print(f"  {exp:<17} - Run {EXPERIMENTS[exp]}")
    print("")
    print("Examples:")
    print("  python experiment_runner.py drift")
    print("  python experiment_runner.py sweep system.num_servers=5 simulation.ssa.sim_time=5000")
    print("  python experiment_runner.py policy +simulation.export_trajectories=True")

def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform execution script for GibbsQ experiments",
        add_help=False
    )
    parser.add_argument("experiment", nargs="?", help="Experiment name to run")
    parser.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Progress bar mode for child experiments (default: auto)",
    )
    
    args, hydra_overrides = parser.parse_known_args()
    
    if not args.experiment or args.experiment in ["-h", "--help", "help"]:
        print_usage()
        return 0 if args.experiment in ["-h", "--help", "help"] else 1
    
    experiment = args.experiment.lower()
    
    if experiment not in EXPERIMENTS:
        print(f"Error: Unknown experiment '{experiment}'")
        print(f"Valid options are: {', '.join(EXPERIMENTS.keys())}")
        return 1
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Set PYTHONPATH
    env = os.environ.copy()
    # Include both project root (for experiments) and src (for gibbsq package).
    # Preserve any existing PYTHONPATH entries to avoid shadowing user env setup.
    existing_pythonpath = env.get("PYTHONPATH")
    path_parts = [str(project_root), str(project_root / "src")]
    if existing_pythonpath:
        path_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    configure_progress_mode(args.progress, env=env)
    
    python_script = EXPERIMENTS[experiment]
    resolved_hydra_overrides = hydra_overrides + default_hydra_overrides_for_experiment(
        experiment,
        hydra_overrides,
    )
    
    print("=" * 58)
    print(f" Starting Experiment: {experiment}")
    print(f" Progress Mode: {args.progress}")
    print(f" Remaining Args (Hydra Overrides): {' '.join(resolved_hydra_overrides)}")
    print("=" * 58)
    
    # Change to project root and run
    os.chdir(project_root)
    
    cmd = [sys.executable, "-m", python_script] + resolved_hydra_overrides
    
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running experiment: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
