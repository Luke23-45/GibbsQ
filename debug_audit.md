# Debug Block Removal Audit

**Date**: 2026-03-27
**Purpose**: Remove debug block implementations from experiment scripts since config files (small/fast) serve as debug runs

## Analysis Summary

### Why Remove Debug Blocks?
- Config files `small.yaml` and `fast.yaml` already provide debug/quick validation runs
- The `debug` flag in code duplicates this functionality unnecessarily
- Removing it simplifies the codebase and relies on config-driven behavior

## Files Scanned
| File | Status | Debug Blocks Found | Action Taken |
|------|--------|-------------------|--------------|
| `experiments/testing/stress_test.py` | ✅ Complete | 5 conditionals | Removed all debug conditionals |
| `src/gibbsq/core/config.py` | ✅ Complete | 3 occurrences | Removed debug fields and cleaned comments |
| `scripts/verification/validation_suite.py` | ✅ Complete | 4 occurrences | Removed debug_flag, renamed level to "quick" |
| `configs/default.yaml` | ✅ Complete | 1 occurrence | Removed `debug: false` line |
| `configs/final_experiment.yaml` | ✅ Complete | 1 occurrence | Removed `debug: false` line |
| `configs/small.yaml` | ✅ Complete | 1 occurrence | Cleaned comment |
| `configs/fast.yaml` | ✅ Complete | 1 occurrence | Cleaned header comment |
| `configs/debug.yaml` | 🔥 Deleted | N/A | Removed redundant config file |
| `tests/baseline_test.py` | ✅ Complete | 2 occurrences | Updated to use `small.yaml` |
| `README.md` | ✅ Complete | 1 occurrence | Removed stale `debug=true` instruction |

### Detailed Findings

#### 1. `experiments/testing/stress_test.py` (5 debug conditionals)
- **Line 109**: `n_targets = [cfg.stress.n_values[0]] if raw_cfg.get("debug", False) else cfg.stress.n_values`
  - When debug=True: uses only first N value
  - When debug=False: uses all N values from config
  - **Resolution**: Always use `cfg.stress.n_values` (config controls this)

- **Line 117**: `_sim_time_t1 = 100.0 if raw_cfg.get("debug", False) else cfg.stress.massive_n_sim_time`
  - When debug=True: hardcoded 100.0s
  - When debug=False: uses `cfg.stress.massive_n_sim_time`
  - **Resolution**: Always use `cfg.stress.massive_n_sim_time`

- **Line 181**: `rho_targets = [cfg.stress.critical_rhos[0]] if raw_cfg.get("debug", False) else cfg.stress.critical_rhos`
  - When debug=True: uses only first rho value
  - When debug=False: uses all rho values from config
  - **Resolution**: Always use `cfg.stress.critical_rhos`

- **Lines 192-204**: `if raw_cfg.get("debug", False): _sim_time_crit = 500.0 else: ...`
  - When debug=True: hardcoded 500.0s
  - When debug=False: uses computed sim_time from config
  - **Resolution**: Always use the else branch logic

- **Line 330**: `_sim_time_het = 500.0 if raw_cfg.get("debug", False) else cfg.stress.heterogeneity_sim_time`
  - When debug=True: hardcoded 500.0s
  - When debug=False: uses `cfg.stress.heterogeneity_sim_time`
  - **Resolution**: Always use `cfg.stress.heterogeneity_sim_time`

#### 2. `src/gibbsq/core/config.py` (2 occurrences)
- **Line 420**: `debug: bool = False` - dataclass field
- **Line 761**: `debug=d.get("debug", False)` - in hydra_to_config
- **Resolution**: Remove the field entirely

#### 3. `scripts/verification/validation_suite.py` (4 occurrences)
- **Line 83**: Removed `debug_flag` definition
- **Line 87**: Removed `f"debug={debug_flag}"` passing to stress test
- **Line 176**: Renamed `choices=["debug", ...]` to `choices=["quick", ...]`
- **Resolution**: Removed ephemeral debug logic; transitioned verification level name to "quick" for clarity.

#### 4. Config Files
- `configs/default.yaml`: Removed `debug: false`
- `configs/final_experiment.yaml`: Removed `debug: false`
- `configs/fast.yaml`: Renamed "Fast debug..." to "Fast validation..." in header
- `configs/debug.yaml`: **DELETED**. Use `small.yaml` or `fast.yaml` for validation.

#### 5. `tests/baseline_test.py`
- Updated `test_rho_max_in_config` and `test_batch_size_in_config` to point to `small.yaml` instead of `debug.yaml`.
- Verified expectations match `small.yaml` (rho_max=0.85, batch_size=64).

### Summary
- **Total Files Scanned**: 10
- **Files Modified**: 7
- **Files Deleted**: 1
- **Debug References Polished**: ~20
- **Lines Removed/Modified**: ~15

---

## Verification
Final grep search confirmed no remaining `raw_cfg.get("debug"` or `^debug:` patterns in the codebase.

### Python Syntax Verification
- ✅ `experiments/testing/stress_test.py` - Compiles successfully
- ✅ `src/gibbsq/core/config.py` - Compiles successfully  
- ✅ `scripts/verification/validation_suite.py` - Compiles successfully

### YAML Syntax Verification
- ✅ `configs/default.yaml` - Valid YAML
- ✅ `configs/final_experiment.yaml` - Valid YAML

### Dataclass Field Verification
```
ExperimentConfig fields: ['system', 'simulation', 'policy', 'drift', 'wandb', 'jax', 
'jax_engine', 'neural', 'verification', 'generalization', 'stress', 'stability_sweep', 
'domain_randomization', 'neural_training', 'output_dir', 'log_dir', 'train_epochs', 'batch_size']
```
**No `debug` field present** ✅

## Impact
- **Before**: Debug behavior controlled by `debug` flag in code + config
- **After**: Debug behavior controlled entirely by config files (`small.yaml`, `fast.yaml`)
- **Benefit**: Cleaner code, single source of truth (config files)

---

## Audit Log
- [2026-03-27] Initial audit of existing uncommitted changes.
- [2026-03-27] Confirmed removal of `debug` fields in `config.py` and conditionals in `stress_test.py`.
- [2026-03-27] Removed `debug_flag` and renamed "debug" level to "quick" in `validation_suite.py`.
- [2026-03-27] Deleted `debug.yaml` as redundant; updated `baseline_test.py` to use `small.yaml`.
- [2026-03-27] Cleaned "debug" terminology from comments in `config.py` and `fast.yaml`.
- [2026-03-27] Removed stale instructions from `README.md`.
- [2026-03-27] Final project-wide grep verification: No functional debug logic remaining. ✅
