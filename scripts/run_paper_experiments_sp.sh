#!/usr/bin/env bash
# Full end-to-end master execution pipeline for the GibbsQ & N-GibbsQ research paper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_experiment.sh"

echo -e "\033[1;35m==========================================================\033[0m"
echo -e "\033[1;35m  GibbsQ Research Paper: Final Execution Pipeline\033[0m"
echo -e "\033[1;35m==========================================================\033[0m"

echo -e "\n\033[1;36m[Initiating Pipeline...]\033[0m\n"

# Parse arguments for step selection and Hydra args
START_STEP=1
STOP_STEP=10
PASSED_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --start|--from)
      START_STEP="$2"
      shift 2
      ;;
    --stop|--to)
      STOP_STEP="$2"
      shift 2
      ;;
    *)
      PASSED_ARGS+=("$1")
      shift
      ;;
  esac
done

# Set positional arguments back to passed args for downstream scripts
set -- "${PASSED_ARGS[@]}"



# ---------------------------------------------------------
# Phase 1: Foundational Analytical Metrics & Verification
# ---------------------------------------------------------
if [[ 1 -ge $START_STEP && 1 -le $STOP_STEP ]]; then
  echo -e "\n\033[1;33m[1/10] Running Drift Verification (Phase 1a)...\033[0m"
  "$RUN_SCRIPT" drift "$@"
fi

if [[ 2 -ge $START_STEP && 2 -le $STOP_STEP ]]; then
  echo -e "\n\033[1;33m[2/10] Running Model Fidelity Check (Phase 1b)...\033[0m"
  "$RUN_SCRIPT" fidelity "$@"
fi

if [[ 3 -ge $START_STEP && 3 -le $STOP_STEP ]]; then
  echo -e "\n\033[1;33m[3/10] Running Jacobian Rigor (AD Check)...\033[0m"
  "$RUN_SCRIPT" jacobian simulation.dga.sim_steps=500 "$@"
fi

if [[ 4 -ge $START_STEP && 4 -le $STOP_STEP ]]; then
  echo -e "\n\033[1;33m[4/10] Running Policy Evaluation Benchmark...\033[0m"
  "$RUN_SCRIPT" policy "+experiment=policy_comparison" "$@"
fi

if [[ 5 -ge $START_STEP && 5 -le $STOP_STEP ]]; then
  echo -e "\n\033[1;33m[5/10] Running Stability Sweep...\033[0m"
  "$RUN_SCRIPT" sweep "+experiment=stability_sweep" "$@"
fi

if [[ 6 -ge $START_STEP && 6 -le $STOP_STEP ]]; then
  echo -e "\n\033[1;33m[6/10] Running Scaling Stress Tests...\033[0m"
  "$RUN_SCRIPT" stress ++jax.enabled=True "$@"
fi

# ---------------------------------------------------------
# Phase 2: N-GibbsQ Neural Learning Pipeline
# ---------------------------------------------------------
if [[ 7 -ge $START_STEP && 7 -le $STOP_STEP ]]; then
  echo -e "\n\033[1;33m[7/10] Running DGA Routing Agent Core Training...\033[0m"
  "$RUN_SCRIPT" train "$@"
fi

if [[ 8 -ge $START_STEP && 8 -le $STOP_STEP ]]; then
  echo -e "\n\033[1;33m[8/10] Running Neural Curriculum Training...\033[0m"
  "$RUN_SCRIPT" n_train "$@"
fi

if [[ 9 -ge $START_STEP && 9 -le $STOP_STEP ]]; then
  echo -e "\n\033[1;33m[9/10] Verifying Neural Parity against GibbsQ Ground Truth...\033[0m"
  "$RUN_SCRIPT" parity "$@"
fi

# ---------------------------------------------------------
# Phase 3: Generational Analysis & Ablation
# ---------------------------------------------------------
if [[ 10 -ge $START_STEP && 10 -le $STOP_STEP ]]; then
  echo -e "\n\033[1;33m[10/10] Running Deep Component Ablation & Generational Generalization...\033[0m"
  "$RUN_SCRIPT" stats "$@"
  "$RUN_SCRIPT" generalize "$@"
  "$RUN_SCRIPT" critical "$@"
  "$RUN_SCRIPT" ablation "$@"
fi

echo -e "\n\033[1;32m==========================================================\033[0m"
echo -e "\033[1;32m  Pipeline fully complete.\033[0m"
echo -e "\033[1;32m  Review '/outputs/' for your plots and logs.\033[0m"
echo -e "\033[1;32m==========================================================\033[0m"
