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
# Phase 0: DGA Bias Quantification (SG#2 FIX)
# Must run BEFORE any training or evaluation step.
# Empirically measures DGA surrogate bias vs true Gillespie SSA.
# ---------------------------------------------------------
echo -e "\n\033[1;33m[0/11] Running DGA Bias Verification (SG#2)...\033[0m"
"$RUN_SCRIPT" bias "$@"

# ---------------------------------------------------------
# Phase 1: Foundational Analytical Metrics & Verification
# ---------------------------------------------------------
echo -e "\n\033[1;33m[1/11] Running Drift Verification (Phase 1a)...\033[0m"
"$RUN_SCRIPT" drift "$@"

echo -e "\n\033[1;33m[2/10] Running Model Fidelity Check (Phase 1b)...\033[0m"
"$RUN_SCRIPT" fidelity "$@"

echo -e "\n\033[1;33m[3/10] Running Jacobian Rigor (AD Check)...\033[0m"
"$RUN_SCRIPT" jacobian simulation.dga.sim_steps=500 "$@"

echo -e "\n\033[1;33m[4/10] Running Policy Evaluation Benchmark...\033[0m"
"$RUN_SCRIPT" policy "+experiment=policy_comparison" "$@"

echo -e "\n\033[1;33m[5/10] Running Stability Sweep...\033[0m"
"$RUN_SCRIPT" sweep "+experiment=stability_sweep" "$@"

echo -e "\n\033[1;33m[6/10] Running Scaling Stress Tests...\033[0m"
"$RUN_SCRIPT" stress ++jax.enabled=True "$@"

# ---------------------------------------------------------
# Phase 2: N-GibbsQ Neural Learning Pipeline
# ---------------------------------------------------------
echo -e "\n\033[1;33m[7/10] Running DGA Routing Agent Core Training...\033[0m"
"$RUN_SCRIPT" train "$@"

echo -e "\n\033[1;33m[8/10] Running Neural Curriculum Training...\033[0m"
"$RUN_SCRIPT" n_train "$@"

echo -e "\n\033[1;33m[9/10] Verifying Neural Parity against GibbsQ Ground Truth...\033[0m"
"$RUN_SCRIPT" parity "$@"

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
