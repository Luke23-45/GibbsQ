#!/usr/bin/env python3
"""
Generate machine-checkable CSV artifacts for the active four-profile config set.

Root-profile rows are derived from the resolved runtime config, not just the
literal YAML surface. This means schema defaults that become active at runtime
are audited even when omitted from the YAML file.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gibbsq.core.config import PROFILE_CONFIG_NAMES, runtime_root_dict

CONFIG_NAMES = list(PROFILE_CONFIG_NAMES)
YAML_FILES = {name: f"{name}.yaml" for name in CONFIG_NAMES}


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        elif v is None:
            items.append((new_key, "null"))
        else:
            items.append((new_key, v))
    return items


def load_yaml_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for name, filename in YAML_FILES.items():
        filepath = config_dir / filename
        if filepath.exists():
            configs[name] = OmegaConf.to_container(OmegaConf.load(filepath), resolve=True)
            print(f"[OK] Loaded {filename}")
        else:
            print(f"[WARN] {filename} not found")
    return configs


def _root_only_yaml(config_data: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v for k, v in config_data.items()
        if k not in {"experiments", "active_experiment", "active_profile"}
    }


def extract_all_parameters(configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    all_params: dict[str, dict[str, Any]] = defaultdict(dict)
    for config_name, config_data in configs.items():
        root_cfg = runtime_root_dict(OmegaConf.create(_root_only_yaml(config_data)))
        for param_path, value in flatten_dict(root_cfg):
            all_params[param_path][config_name] = value

        experiments = config_data.get("experiments", {})
        if isinstance(experiments, dict):
            for param_path, value in flatten_dict({"experiments": experiments}):
                all_params[param_path][config_name] = value
    return all_params


def strip_experiment_prefix(param_path: str) -> tuple[str | None, str]:
    parts = param_path.split(".")
    if len(parts) >= 3 and parts[0] == "experiments":
        return parts[1], ".".join(parts[2:])
    return None, param_path


def classify_base_parameter(param_path: str) -> tuple[str, str, str]:
    exact: dict[str, tuple[str, str, str]] = {
        "output_dir": ("Metadata/Output", "Base output directory for run artifacts.", "Profile-specific output identity; may vary by profile."),
        "log_dir": ("Metadata/Output", "Derived log directory for the profile outputs.", "Should track output_dir, not scientific semantics."),
        "train_epochs": ("Profile-Scaled", "Training epoch budget for the main training loop.", "May increase with profile rigor."),
        "batch_size": ("Profile-Scaled", "Mini-batch size for training/evaluation routines that consume it.", "May increase with profile rigor."),
        "system.num_servers": ("Workload-Defining", "Number of parallel servers in the modeled queueing system.", "Only vary when the profile intentionally changes the modeled queueing regime."),
        "system.arrival_rate": ("Workload-Defining", "Poisson arrival rate lambda of the modeled system.", "Only vary when the profile intentionally changes the modeled queueing regime."),
        "system.service_rates": ("Workload-Defining", "Per-server service rates of the modeled system.", "Only vary when the profile intentionally changes the modeled queueing regime."),
        "system.alpha": ("Invariant", "Inverse temperature alpha used by the routing policy.", "Keep fixed unless intentionally changing the theoretical policy regime."),
        "simulation.num_replications": ("Profile-Scaled", "Number of independent Monte Carlo replications.", "Scale upward with rigor and statistical confidence."),
        "simulation.seed": ("Invariant", "Base random seed for reproducibility.", "Keep constant across profiles unless intentionally reseeding experiments."),
        "simulation.burn_in_fraction": ("Invariant", "Fraction of each trajectory discarded as burn-in.", "Keep fixed unless the stationarity protocol changes."),
        "simulation.export_trajectories": ("Profile-Scaled", "Whether raw trajectories are persisted to disk.", "Enable for final/publication artifact retention only when needed."),
        "simulation.ssa.sim_time": ("Profile-Scaled", "Continuous-time SSA horizon per trajectory.", "Increase with rigor or when the experiment needs longer convergence."),
        "simulation.ssa.sample_interval": ("Invariant", "Sampling interval for SSA trajectory snapshots.", "Keep fixed unless measurement resolution changes."),
        "simulation.dga.sim_steps": ("Profile-Scaled", "Number of differentiable approximation steps.", "Increase with rigor when DGA-backed routines need more resolution."),
        "simulation.dga.temperature": ("Invariant", "Relaxation temperature for differentiable event selection.", "Keep fixed unless the DGA objective changes."),
        "policy.name": ("Invariant", "Root routing policy family used outside experiment-specific overrides.", "Keep fixed unless changing the root policy semantics."),
        "policy.d": ("Invariant", "Power-of-d branching parameter when applicable.", "Keep fixed unless the routing policy family changes."),
        "drift.q_max": ("Profile-Scaled", "Queue grid upper bound for drift verification.", "Increase with rigor when larger compact-set coverage is required."),
        "drift.use_grid": ("Invariant", "Whether drift verification uses grid evaluation.", "Keep fixed unless the drift evaluation method changes."),
        "stress.n_values": ("Profile-Scaled", "Server-count grid used by scaling stress experiments.", "Increase with profile rigor."),
        "stress.critical_rhos": ("Profile-Scaled", "Load factors examined in critical-load stress runs.", "Increase breadth and extremity with rigor."),
        "stress.mu_het": ("Invariant", "Canonical heterogeneous service-rate vector for heterogeneity stress tests.", "Keep fixed so heterogeneity comparisons remain comparable."),
        "stress.sample_interval": ("Invariant", "Sampling interval for stress trajectories.", "Keep fixed unless measurement resolution changes."),
        "stress.massive_n_rho": ("Invariant", "Target load factor for massive-N stress runs.", "Keep fixed so scale comparisons stay comparable."),
        "stress.massive_n_sim_time": ("Invariant", "Trajectory horizon for massive-N stress runs.", "Keep fixed unless the stress protocol changes."),
        "stress.critical_load_n": ("Invariant", "Server count used in critical-load stress experiments.", "Keep fixed unless the benchmark definition changes."),
        "stress.critical_load_base_rho": ("Invariant", "Base load used by critical-load horizon scaling.", "Keep fixed unless the scaling law changes."),
        "stress.critical_load_max_sim_time": ("Profile-Scaled", "Maximum allowed horizon for critical-load experiments.", "Increase with rigor to avoid truncating near-capacity runs."),
        "stress.heterogeneity_rho": ("Invariant", "Target load factor for heterogeneity experiments.", "Keep fixed so heterogeneity comparisons stay comparable."),
        "stress.heterogeneity_sim_time": ("Invariant", "Trajectory horizon for heterogeneity experiments.", "Keep fixed unless the heterogeneity protocol changes."),
        "generalization.rho_boundary_vals": ("Profile-Scaled", "Boundary-load values used by generalization evaluations.", "Increase coverage with rigor."),
        "generalization.scale_vals": ("Profile-Scaled", "Problem-scale multipliers used in generalization sweeps.", "Increase coverage with rigor."),
        "generalization.rho_grid_vals": ("Profile-Scaled", "Grid of load factors used in generalization heatmaps.", "Increase coverage with rigor."),
        "stability_sweep.alpha_vals": ("Profile-Scaled", "Alpha sweep used by stability exploration.", "Increase breadth with rigor."),
        "stability_sweep.rho_vals": ("Profile-Scaled", "Rho sweep used by stability exploration.", "Increase breadth with rigor."),
        "neural.hidden_size": ("Workload-Defining", "Width of the neural routing/value networks.", "May vary when the profile intentionally changes model capacity."),
        "neural.preprocessing": ("Workload-Defining", "Input preprocessing applied before neural inference.", "Change only when the intended model architecture changes."),
        "neural.capacity_bound": ("Invariant", "Capacity normalization bound used by neural feature scaling.", "Keep fixed unless the neural feature contract changes."),
        "neural.init_type": ("Workload-Defining", "Network initialization regime.", "Change only when the intended model architecture changes."),
        "neural.use_rho": ("Invariant", "Whether load factor rho is fed to the neural model.", "Keep fixed unless the model input contract changes."),
        "neural.use_service_rates": ("Invariant", "Whether service rates are fed to the neural model.", "Keep fixed unless the model input contract changes."),
        "neural.rho_input_scale": ("Workload-Defining", "Scaling applied to the rho input feature.", "Change only when the intended feature encoding changes."),
        "neural.entropy_bonus": ("Invariant", "Base entropy regularization coefficient.", "Keep fixed unless the optimization objective changes."),
        "neural.entropy_final": ("Invariant", "Final-stage entropy regularization coefficient.", "Keep fixed unless the optimization objective changes."),
        "neural.clip_global_norm": ("Invariant", "Gradient clipping threshold for neural optimization.", "Keep fixed unless optimization stability policy changes."),
        "neural.actor_lr": ("Workload-Defining", "Actor network learning rate.", "Change only when the intended optimization regime changes."),
        "neural.critic_lr": ("Workload-Defining", "Critic/value network learning rate.", "Change only when the intended optimization regime changes."),
        "neural.lr_decay_rate": ("Invariant", "Learning-rate decay factor.", "Keep fixed unless scheduler semantics change."),
        "neural.weight_decay": ("Invariant", "Weight decay used by neural optimizers.", "Keep fixed unless optimization regularization changes."),
        "neural_training.learning_rate": ("Invariant", "Top-level neural training learning rate.", "Keep fixed unless the training objective changes."),
        "neural_training.dga_learning_rate": ("Invariant", "Learning rate used by DGA-backed training routines.", "Keep fixed unless the training objective changes."),
        "neural_training.weight_decay": ("Invariant", "Weight decay used by training routines.", "Keep fixed unless regularization policy changes."),
        "neural_training.min_temperature": ("Invariant", "Minimum exploration or relaxation temperature during training.", "Keep fixed unless training semantics change."),
        "neural_training.gamma": ("Invariant", "Discount factor used by RL routines.", "Keep fixed unless the return definition changes."),
        "neural_training.gae_lambda": ("Invariant", "GAE lambda used by RL routines.", "Keep fixed unless the estimator changes."),
        "neural_training.curriculum": ("Profile-Scaled", "Curriculum phases controlling horizon and epoch growth.", "Scale with profile rigor while preserving intended progression."),
        "neural_training.eval_batches": ("Profile-Scaled", "Number of evaluation batches during training.", "Increase with profile rigor."),
        "neural_training.eval_trajs_per_batch": ("Profile-Scaled", "Trajectories per evaluation batch.", "Increase with profile rigor."),
        "neural_training.bc_num_steps": ("Profile-Scaled", "Behavior-cloning optimization step budget.", "Increase with profile rigor."),
        "neural_training.bc_lr": ("Invariant", "Behavior-cloning learning rate.", "Keep fixed unless the training objective changes."),
        "neural_training.bc_label_smoothing": ("Invariant", "Behavior-cloning label smoothing strength.", "Keep fixed unless the training objective changes."),
        "neural_training.perf_index_min_denom": ("Invariant", "Minimum denominator for performance-index normalization.", "Keep fixed unless metric semantics change."),
        "neural_training.perf_index_jsq_margin": ("Invariant", "JSQ margin used in performance-index normalization.", "Keep fixed unless metric semantics change."),
        "neural_training.shake_scale": ("Invariant", "Perturbation scale used by gradient-shake routines.", "Keep fixed unless estimator semantics change."),
        "neural_training.checkpoint_freq": ("Profile-Scaled", "Checkpoint cadence during training.", "May vary with profile budget and artifact needs."),
        "neural_training.squash_scale": ("Invariant", "Scale factor for target squashing.", "Keep fixed unless target transformation semantics change."),
        "neural_training.squash_threshold": ("Invariant", "Threshold for target squashing.", "Keep fixed unless target transformation semantics change."),
        "domain_randomization.enabled": ("Workload-Defining", "Whether domain randomization is used during training.", "Enable only when the profile intends broader training generalization."),
        "domain_randomization.rho_min": ("Workload-Defining", "Minimum randomized load factor used during training.", "Change only when the intended training regime changes."),
        "domain_randomization.rho_max": ("Workload-Defining", "Maximum randomized load factor used during training.", "Change only when the intended training regime changes."),
        "domain_randomization.phases": ("Workload-Defining", "Explicit domain-randomization curriculum phases active at runtime.", "Treat omission carefully because runtime may inject or suppress defaults by policy."),
        "jax.enabled": ("Profile-Scaled", "Whether JAX acceleration is enabled at the root profile level.", "Enable as needed for runtime budget or target hardware."),
        "jax.platform": ("Invariant", "Requested JAX backend selection policy.", "Keep fixed unless device-selection semantics change."),
        "jax.precision": ("Invariant", "Requested JAX floating-point precision.", "Keep fixed unless a numerics contract changes."),
        "jax.fallback_to_cpu": ("Invariant", "Whether JAX may fall back to CPU when the target platform is unavailable.", "Keep fixed unless failure-handling semantics change."),
        "jax_engine.max_events_safety_multiplier": ("Invariant", "Safety multiplier for JAX event-count budgeting.", "Keep fixed unless the engine safety contract changes."),
        "jax_engine.max_events_additive_buffer": ("Invariant", "Additive safety buffer for JAX event-count budgeting.", "Keep fixed unless the engine safety contract changes."),
        "jax_engine.scan_sampling_chunk": ("Invariant", "Chunk size used by JAX scan-based sampling.", "Keep fixed unless the engine sampling contract changes."),
        "verification.parity_threshold_percent": ("Invariant", "Allowed parity margin for verification checks.", "Keep fixed unless success criteria change."),
        "verification.jacobian_rel_tol": ("Invariant", "Relative tolerance for Jacobian and gradient verification.", "Keep fixed unless verification criteria change."),
        "verification.alpha_significance": ("Invariant", "Alpha level for statistical significance tests.", "Keep fixed unless verification criteria change."),
        "verification.confidence_interval": ("Invariant", "Confidence level used by reporting and verification intervals.", "Keep fixed unless verification criteria change."),
        "verification.stationarity_threshold": ("Invariant", "Required fraction of replications meeting stationarity.", "Keep fixed unless certification criteria change."),
        "verification.parity_z_score": ("Invariant", "Z-score used in parity-style statistical bounds.", "Keep fixed unless verification criteria change."),
        "verification.gradient_check_chunk_size": ("Profile-Scaled", "Chunk size used during gradient verification.", "Increase with profile rigor."),
        "verification.gradient_check_max_steps": ("Profile-Scaled", "Maximum optimization or evaluation steps used in gradient checks.", "Increase with profile rigor."),
        "verification.gradient_check_n_test": ("Invariant", "Number of test points used in gradient checks.", "Keep fixed unless the verification protocol changes."),
        "verification.gradient_check_hidden_size": ("Profile-Scaled", "Optional hidden-size override used in gradient checks.", "Only vary when the verification workload intentionally changes."),
        "verification.gradient_check_sim_time": ("Profile-Scaled", "Simulation horizon used inside gradient checks.", "Increase with profile rigor."),
        "verification.gradient_check_n_samples": ("Profile-Scaled", "Number of Monte Carlo samples used in gradient checks.", "Increase with profile rigor."),
        "verification.gradient_check_epsilon": ("Invariant", "Finite-difference epsilon used in gradient checks.", "Keep fixed unless estimator semantics change."),
        "verification.gradient_check_cosine_threshold": ("Invariant", "Minimum cosine similarity accepted by gradient verification.", "Keep fixed unless success criteria change."),
        "verification.gradient_check_error_threshold": ("Profile-Scaled", "Maximum allowed relative error in gradient verification.", "May vary between smoke and research-grade checks."),
        "verification.gradient_shake_scale": ("Invariant", "Perturbation scale used in gradient robustness checks.", "Keep fixed unless verification semantics change."),
        "wandb.enabled": ("Metadata/Output", "Whether W&B logging is enabled.", "Operational metadata; may vary by execution environment."),
        "wandb.project": ("Metadata/Output", "W&B project name used for logging runs.", "Operational metadata; should reflect the intended profile namespace."),
        "wandb.entity": ("Metadata/Output", "Optional W&B entity/account name.", "Operational metadata; keep stable unless logging ownership changes."),
        "wandb.group": ("Metadata/Output", "Optional W&B grouping label at the root profile level.", "Operational metadata; may vary by run organization policy."),
        "wandb.tags": ("Metadata/Output", "Optional W&B tags attached to runs.", "Operational metadata; may vary by reporting needs."),
        "wandb.mode": ("Metadata/Output", "W&B online or offline mode.", "Operational metadata; may vary by execution environment."),
        "wandb.run_name": ("Metadata/Output", "Explicit W&B run name.", "Operational metadata; may vary by profile or run intent."),
    }
    if param_path in exact:
        return exact[param_path]
    prefix = param_path.split(".")[0]
    return (
        "Metadata/Output" if prefix == "wandb" else "Workload-Defining",
        f"Parameter within the {prefix} configuration family.",
        f"Review {prefix} family policy when changing {param_path}.",
    )


def classify_parameter(param_path: str) -> tuple[str, str, str]:
    experiment_name, base_param = strip_experiment_prefix(param_path)
    if experiment_name is not None:
        _, base_meaning, _ = classify_base_parameter(base_param)
        return (
            "Experiment-Specific Override",
            f"{experiment_name} override of {base_meaning[0].lower() + base_meaning[1:] if base_meaning else base_param}.",
            "Only valid when this experiment's semantics intentionally diverge from the profile root.",
        )
    return classify_base_parameter(base_param)


def write_csv(output_path: Path, header: list[str], rows: list[list[Any]], label: str) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[OK] Written {label}: {output_path}")


def write_outputs(config_dir: Path, filename: str, header: list[str], rows: list[list[Any]], label: str) -> None:
    output_path = config_dir / filename
    write_csv(output_path, header, rows, label)


def build_comprehensive_rows(all_params: dict[str, dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for param_path in sorted(all_params.keys(), key=lambda x: (x.count("."), x)):
        rows.append([param_path] + [all_params[param_path].get(config_name, "N/A") for config_name in CONFIG_NAMES])
    return rows


def build_summary_rows(all_params: dict[str, dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for param_path in sorted(all_params.keys(), key=lambda x: (x.count("."), x)):
        values = [str(all_params[param_path].get(config_name, "N/A")) for config_name in CONFIG_NAMES]
        if len(set(values)) > 1:
            rows.append([param_path] + [all_params[param_path].get(config_name, "N/A") for config_name in CONFIG_NAMES])
    return rows


def build_category_rows(all_params: dict[str, dict[str, Any]]) -> list[list[Any]]:
    categories: dict[str, list[str]] = defaultdict(list)
    for param_path in all_params.keys():
        categories[param_path.split(".")[0]].append(param_path)
    rows: list[list[Any]] = []
    for category in sorted(categories.keys()):
        for param_path in sorted(categories[category], key=lambda x: (x.count("."), x)):
            rows.append([category, param_path] + [all_params[param_path].get(config_name, "N/A") for config_name in CONFIG_NAMES])
    return rows


def build_parameter_ledger_rows(all_params: dict[str, dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for param_path in sorted(all_params.keys(), key=lambda x: (x.count("."), x)):
        classification, meaning, decision_rule = classify_parameter(param_path)
        rows.append(
            [param_path.split(".")[0], param_path, classification, meaning, decision_rule]
            + [all_params[param_path].get(config_name, "N/A") for config_name in CONFIG_NAMES]
        )
    return rows


def verify_completeness(configs: dict[str, dict[str, Any]], all_params: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    for config_name, config_data in configs.items():
        root_runtime = runtime_root_dict(OmegaConf.create(_root_only_yaml(config_data)))
        expected_root_paths = {param_path for param_path, _ in flatten_dict(root_runtime)}
        actual_root_paths = {
            path for path, profile_values in all_params.items()
            if not path.startswith("experiments.") and config_name in profile_values
        }
        print(f"\n{config_name}.yaml:")
        print(f"  Resolved root parameters: {len(expected_root_paths)}")
        missing_root = sorted(expected_root_paths - actual_root_paths)
        if missing_root:
            print(f"  [FAIL] Missing resolved root parameters: {missing_root[:10]}")
        else:
            print("  [OK] All resolved root parameters captured")
    print("\n" + "=" * 60)


def main() -> None:
    notes_dir = Path(__file__).parent
    config_dir = notes_dir.parent
    print("=" * 60)
    print("CONFIG PARAMETER COMPARISON TOOL")
    print("=" * 60)
    print(f"\nConfig directory: {config_dir}")
    print(f"Notes directory: {notes_dir}\n")

    print("Loading YAML configs...")
    configs = load_yaml_configs(config_dir)
    if len(configs) < len(CONFIG_NAMES):
        print("\n[FAIL] Error: Not all config files found!")
        return

    print("\nExtracting parameters...")
    all_params = extract_all_parameters(configs)
    verify_completeness(configs, all_params)

    print("\nGenerating CSV files...")
    print("-" * 60)
    write_outputs(
        notes_dir,
        "config_comparison_comprehensive.csv",
        ["Parameter"] + CONFIG_NAMES,
        build_comprehensive_rows(all_params),
        "comprehensive CSV",
    )
    write_outputs(
        notes_dir,
        "config_comparison_summary.csv",
        ["Parameter"] + CONFIG_NAMES,
        build_summary_rows(all_params),
        "summary CSV",
    )
    write_outputs(
        notes_dir,
        "config_comparison_by_category.csv",
        ["Category", "Parameter"] + CONFIG_NAMES,
        build_category_rows(all_params),
        "category-grouped CSV",
    )
    write_outputs(
        notes_dir,
        "parameter_freeze_ledger.csv",
        ["Category", "Parameter", "Classification", "Meaning", "DecisionRule"] + CONFIG_NAMES,
        build_parameter_ledger_rows(all_params),
        "parameter freeze ledger",
    )

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated CSV files in: {notes_dir}")
    print("  1. config_comparison_comprehensive.csv - Resolved root parameters plus explicit experiment overrides")
    print("  2. config_comparison_summary.csv - Only differing parameters")
    print("  3. config_comparison_by_category.csv - Grouped by category")
    print("  4. parameter_freeze_ledger.csv - Parameter classification and freeze rules")


if __name__ == "__main__":
    main()
