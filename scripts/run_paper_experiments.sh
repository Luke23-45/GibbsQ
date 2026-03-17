#!/usr/bin/env bash
# Full end-to-end master execution pipeline for the GibbsQ & N-GibbsQ research paper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_experiment.sh"

echo -e "\033[1;35m==========================================================\033[0m"
echo -e "\033[1;35m  GibbsQ Research Paper: Final Execution Pipeline\033[0m"
echo -e "\033[1;35m==========================================================\033[0m"

echo -e "\n\033[1;36m[Initiating Pipeline...]\033[0m\n"


# ---------------------------------------------------------
# Phase 0: Gradient Estimator Validation (Track 5)
# Must run BEFORE any training to ensure unbiased gradients.
# ---------------------------------------------------------
echo -e "\n\033[1;33m[0/10] Running REINFORCE Gradient validation (Track 5)...\033[0m"
"$RUN_SCRIPT" reinforce_check "$@"

# ---------------------------------------------------------
# Phase 1: Foundational Analytical Metrics & Verification
# ---------------------------------------------------------
echo -e "\n\033[1;33m[1/11] Running Drift Verification (Phase 1a)...\033[0m"
"$RUN_SCRIPT" drift "$@"

echo -e "\n\033[1;33m[2/10] Running Model Fidelity Check (Phase 1b)...\033[0m"
"$RUN_SCRIPT" fidelity "$@"

echo -e "\n\033[1;33m[4/10] Running Corrected Policy Evaluation Benchmark (Track 4)...\033[0m"
"$RUN_SCRIPT" corrected_policy "$@"

echo -e "\n\033[1;33m[5/10] Running Stability Sweep...\033[0m"
"$RUN_SCRIPT" sweep "+experiment=stability_sweep" "$@"

echo -e "\n\033[1;33m[6/10] Running Scaling Stress Tests...\033[0m"
"$RUN_SCRIPT" stress ++jax.enabled=True "$@"

# ---------------------------------------------------------
# Phase 2: N-GibbsQ Neural Learning Pipeline
# ---------------------------------------------------------
echo -e "\n\033[1;33m[7/10] Running REINFORCE SSA Training (Track 1)...\033[0m"
"$RUN_SCRIPT" reinforce_train "$@"

echo -e "\n\033[1;33m[8/10] Running Domain Randomization Training (Track 3)...\033[0m"
"$RUN_SCRIPT" dr_train "$@"

# ---------------------------------------------------------
# Phase 3: Generational Analysis & Ablation
# ---------------------------------------------------------
echo -e "\n\033[1;33m[10/10] Running Deep Component Ablation & Generational Generalization...\033[0m"
"$RUN_SCRIPT" stats "$@"
"$RUN_SCRIPT" generalize "$@"
"$RUN_SCRIPT" critical "$@"
"$RUN_SCRIPT" ablation "$@"

echo -e "\n\033[1;32m==========================================================\033[0m"
echo -e "\033[1;32m  Pipeline fully complete.\033[0m"
echo -e "\033[1;32m  Review '/outputs/' for your plots and logs.\033[0m"
echo -e "\033[1;32m==========================================================\033[0m"
