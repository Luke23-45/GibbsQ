# Smoking Gun Registry

## Confirmed Findings

SMOKING GUN #1
  Experiment    : Track 5 (reinforce_check launcher path)
  File          : scripts/run_experiment.sh
  Function      : shell bootstrap / PYTHONPATH export
  Lines         : export PYTHONPATH assignment
  Observed      : PYTHONPATH only included project root, not src; importing gibbsq failed in this environment with that layout.
  Source        : Source 1 = scripts/run_experiment.sh (pre-fix line exporting only $PROJECT_ROOT); Source 2 = pyproject.toml [tool.setuptools.packages.find] where=["src"], plus runtime verification command `PYTHONPATH=/workspace/GibbsQ python -c "import gibbsq"` raising ModuleNotFoundError before fix.
  Impact        : Linux/macOS launcher can fail before running any Phase IV track because experiment modules import `gibbsq.*`.
  Severity      : CRITICAL
  Severity basis: Entry-point failure blocks experiment execution.
  Status        : CONFIRMED (fixed)

SMOKING GUN #2
  Experiment    : Track 3 (dr_train) → Track 4 dependency
  File          : experiments/n_gibbsq/train_domain_randomized.py
  Function      : DomainRandomizedTrainer.execute (pointer write section)
  Lines         : pointer_dir assignment and pointer file writes
  Observed      : Pointer path was hardcoded to outputs/small regardless of active config/output_dir.
  Source        : Source 1 = train_domain_randomized.py hardcoded `_PROJECT_ROOT / "outputs" / "small"`; Source 2 = config system supports configurable `output_dir` (`src/gibbsq/core/config.py` + `configs/default.yaml`) and corrected_policy expects pointer lookup during evaluation.
  Impact        : Track 4 can miss DR weights when run with config output_dir != outputs/small.
  Severity      : HIGH
  Severity basis: Cross-experiment artifact handoff can silently break evaluation fidelity.
  Status        : CONFIRMED (fixed)

SMOKING GUN #3
  Experiment    : Track 4 (corrected_policy)
  File          : experiments/evaluation/corrected_policy_comparison.py
  Function      : run_corrected_comparison (pointer discovery section)
  Lines         : pointer_dir assignment
  Observed      : Pointer lookup path was hardcoded to outputs/small instead of run output root.
  Source        : Source 1 = corrected_policy_comparison.py hardcoded `_PROJECT_ROOT / "outputs" / "small"`; Source 2 = train_reinforce writes pointer relative to configured output root via `self.run_dir.parent.parent`.
  Impact        : Corrected policy benchmark may fail to find model from Track 1/3 in non-small runs.
  Severity      : HIGH
  Severity basis: Track 4 neural parity result can be absent or stale due to wrong pointer directory.
  Status        : CONFIRMED (fixed)

## Summary Table
  | # | Experiment | File | Severity | Status |
  |---|------------|------|----------|--------|
  | 1 | reinforce_check (launcher) | scripts/run_experiment.sh | CRITICAL | CONFIRMED (fixed) |
  | 2 | dr_train | experiments/n_gibbsq/train_domain_randomized.py | HIGH | CONFIRMED (fixed) |
  | 3 | corrected_policy | experiments/evaluation/corrected_policy_comparison.py | HIGH | CONFIRMED (fixed) |

## Cross-Experiment Findings
- SG#2 and SG#3 jointly implicate Track 1/3 → Track 4 artifact handoff.

## Total confirmed : 3
## Total critical  : 1
## Total high      : 2
## Total medium    : 0
