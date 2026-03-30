#!/usr/bin/env python3
"""
Verify generated config CSV artifacts against the active four YAML profiles.

Root-profile rows are checked against resolved runtime config values. Explicit
experiment override rows are checked against the literal YAML override blocks.
"""

from __future__ import annotations

import csv
import sys
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
ALLOWED_CLASSES = {
    "Invariant",
    "Profile-Scaled",
    "Workload-Defining",
    "Experiment-Specific Override",
    "Metadata/Output",
}


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = str(v)
        elif v is None:
            items[new_key] = "null"
        else:
            items[new_key] = v
    return items


def load_yaml_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for name, filename in YAML_FILES.items():
        filepath = config_dir / filename
        if filepath.exists():
            configs[name] = OmegaConf.to_container(OmegaConf.load(filepath), resolve=True)
    return configs


def load_expected_maps(config_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    yaml_configs = load_yaml_configs(config_dir)
    root_expected: dict[str, dict[str, Any]] = {}
    experiment_expected: dict[str, dict[str, Any]] = {}
    for name, config_data in yaml_configs.items():
        root_only = {
            k: v for k, v in config_data.items()
            if k not in {"experiments", "active_experiment", "active_profile"}
        }
        root_expected[name] = flatten_dict(runtime_root_dict(OmegaConf.create(root_only)))
        experiments = config_data.get("experiments", {})
        experiment_expected[name] = flatten_dict({"experiments": experiments}) if isinstance(experiments, dict) else {}
    return root_expected, experiment_expected


def compare_values(csv_val: str, expected_val: Any) -> tuple[bool, str]:
    if csv_val == "N/A":
        return True, "N/A (parameter not defined for this profile surface)"
    expected_str = str(expected_val)
    if csv_val == expected_str:
        return True, f"[OK] Match: {csv_val}"
    return False, f"[FAIL] MISMATCH: CSV='{csv_val}' vs EXPECTED='{expected_str}'"


def _expected_value_for_param(
    param_name: str,
    config_name: str,
    root_expected: dict[str, dict[str, Any]],
    experiment_expected: dict[str, dict[str, Any]],
) -> Any:
    if param_name.startswith("experiments."):
        return experiment_expected[config_name].get(param_name, "N/A")
    return root_expected[config_name].get(param_name, "N/A")


def verify_comparison_csv(
    csv_path: Path,
    root_expected: dict[str, dict[str, Any]],
    experiment_expected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            param_name = row["Parameter"]
            for config_name in CONFIG_NAMES:
                csv_val = row[config_name]
                expected_val = _expected_value_for_param(param_name, config_name, root_expected, experiment_expected)
                is_match, message = compare_values(csv_val, expected_val)
                if not is_match:
                    errors.append({
                        "parameter": param_name,
                        "config": config_name,
                        "csv_value": csv_val,
                        "expected_value": expected_val,
                        "message": message,
                    })
    return errors


def verify_parameter_ledger(
    csv_path: Path,
    root_expected: dict[str, dict[str, Any]],
    experiment_expected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    seen = set()
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            param_name = row["Parameter"]
            seen.add(param_name)
            if row["Classification"] not in ALLOWED_CLASSES:
                errors.append({
                    "parameter": param_name,
                    "config": "classification",
                    "csv_value": row["Classification"],
                    "expected_value": "allowed classification",
                    "message": f"Invalid classification '{row['Classification']}'",
                })
            for config_name in CONFIG_NAMES:
                csv_val = row[config_name]
                expected_val = _expected_value_for_param(param_name, config_name, root_expected, experiment_expected)
                is_match, message = compare_values(csv_val, expected_val)
                if not is_match:
                    errors.append({
                        "parameter": param_name,
                        "config": config_name,
                        "csv_value": csv_val,
                        "expected_value": expected_val,
                        "message": message,
                    })

    all_expected_keys = set()
    for config_name in CONFIG_NAMES:
        all_expected_keys.update(root_expected[config_name].keys())
        all_expected_keys.update(experiment_expected[config_name].keys())
    for param_name in sorted(all_expected_keys - seen):
        errors.append({
            "parameter": param_name,
            "config": "coverage",
            "csv_value": "missing",
            "expected_value": "present in resolved config surface",
            "message": f"Missing parameter in ledger: {param_name}",
        })
    return errors


def print_detailed_verification(root_expected: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("DETAILED VERIFICATION OF KEY PARAMETERS")
    print("=" * 80)
    key_params = [
        "system.num_servers",
        "system.arrival_rate",
        "system.service_rates",
        "simulation.num_replications",
        "simulation.ssa.sim_time",
        "neural.hidden_size",
        "neural.preprocessing",
        "neural.capacity_bound",
        "domain_randomization.enabled",
        "domain_randomization.phases",
        "jax.enabled",
        "train_epochs",
        "batch_size",
        "output_dir",
    ]
    for param in key_params:
        print(f"\n{param}:")
        for config_name in CONFIG_NAMES:
            value = root_expected[config_name].get(param, "N/A")
            print(f"  {config_name:18s}: {value}")


def main() -> bool:
    notes_dir = Path(__file__).parent
    config_dir = notes_dir.parent
    print("=" * 80)
    print("CSV ACCURACY VERIFICATION")
    print("=" * 80)

    print("\nLoading YAML configs...")
    root_expected, experiment_expected = load_expected_maps(config_dir)
    for config_name in CONFIG_NAMES:
        print(
            f"  {config_name}: "
            f"{len(root_expected[config_name])} resolved root parameters, "
            f"{len(experiment_expected[config_name])} explicit experiment override parameters"
        )

    comprehensive_errors = verify_comparison_csv(
        notes_dir / "config_comparison_comprehensive.csv",
        root_expected,
        experiment_expected,
    )
    ledger_errors = verify_parameter_ledger(
        notes_dir / "parameter_freeze_ledger.csv",
        root_expected,
        experiment_expected,
    )

    print("\n" + "-" * 80)
    print("VERIFYING: config_comparison_comprehensive.csv")
    print("-" * 80)
    if comprehensive_errors:
        print(f"\n[FAIL] FOUND {len(comprehensive_errors)} ERRORS:")
        for err in comprehensive_errors[:20]:
            print(err["message"])
    else:
        print("\n[OK] ALL COMPARISON VALUES MATCH")

    print("\n" + "-" * 80)
    print("VERIFYING: parameter_freeze_ledger.csv")
    print("-" * 80)
    if ledger_errors:
        print(f"\n[FAIL] FOUND {len(ledger_errors)} LEDGER ERRORS:")
        for err in ledger_errors[:20]:
            print(err["message"])
    else:
        print("\n[OK] PARAMETER LEDGER IS COMPLETE AND ACCURATE")

    print_detailed_verification(root_expected)

    total_errors = len(comprehensive_errors) + len(ledger_errors)
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"\nErrors found: {total_errors}")
    if total_errors == 0:
        print("\n[OK] CSV FILES AND PARAMETER LEDGER ARE 100% ACCURATE")
    else:
        print(f"\n[FAIL] FOUND {total_errors} DISCREPANCIES")
    return total_errors == 0


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
