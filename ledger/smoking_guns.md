# Smoking Gun Registry

## Confirmed Findings
SMOKING GUN #1
  Experiment    : reinforce_check
  File          : experiments/testing/reinforce_gradient_check.py
  Function      : main
  Lines         : 593-597
  Observed      : When the gradient check fails (`result.passed == False`), the script only logs a warning and does not exit non-zero.
  Source        : (1) experiments/testing/reinforce_gradient_check.py:593-597; (2) scripts/execution/experiment_runner.py:91-92 + scripts/execution/reproduction_pipeline.py:23-27,107-110 (pipeline failure handling is return-code based).
  Impact        : A failed REINFORCE gradient-validity gate can be reported as successful by the orchestrators, allowing downstream phases to proceed on invalid estimator checks.
  Severity      : HIGH
  Severity basis: This is a control-gate integrity failure in the validation phase, not just a metric drift.
  Status        : CONFIRMED

## Summary Table
| # | Experiment | File | Severity | Status |
|---|---|---|---|---|
| 1 | reinforce_check | experiments/testing/reinforce_gradient_check.py | HIGH | CONFIRMED |

## Cross-Experiment Findings
SMOKING GUN #1 impacts the full pipeline because `reproduction_pipeline.py` trusts subprocess exit codes to detect failures.

## Total confirmed : 1
## Total critical  : 0
## Total high      : 1
## Total medium    : 0
